/************************************************************************

    comb.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018 Chad Page
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2020-2021 Adam Sampson
    Copyright (C) 2021 Phillip Blucas

    This file is part of ld-decode-tools.

    ld-chroma-decoder is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#include "comb.h"

#include "framecanvas.h"

#include "deemp.h"
#include "firfilter.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>
#include <QMap>

#include <fftw3.h>
#include <onnxruntime_cxx_api.h>


// Indexes for the candidates considered in 3D adaptive mode
enum CandidateIndex : qint32 {
    CAND_LEFT,
    CAND_RIGHT,
    CAND_UP,
    CAND_DOWN,
    CAND_PREV_FIELD,
    CAND_NEXT_FIELD,
    CAND_PREV_FRAME,
    CAND_NEXT_FRAME,
    NUM_CANDIDATES
};

// Map colours for the candidates
static constexpr quint32 CANDIDATE_SHADES[] = {
    0xFF8080, // CAND_LEFT - red
    0xFF8080, // CAND_RIGHT - red
    0xFFFF80, // CAND_UP - yellow
    0xFFFF80, // CAND_DOWN - yellow
    0x80FF80, // CAND_PREV_FIELD - green
    0x80FF80, // CAND_NEXT_FIELD - green
    0x8080FF, // CAND_PREV_FRAME - blue
    0xFF80FF, // CAND_NEXT_FRAME - purple
};

// Since we are at exactly 4fsc, calculating the value of a in-phase sine wave at a specific position
// is very simple.
static constexpr double sin4fsc_data[] = {1.0, 0.0, -1.0, 0.0};

// 4fsc sine wave
constexpr double sin4fsc(const qint32 i) {
    return sin4fsc_data[i % 4];
}

// 4fsc cos wave
constexpr double cos4fsc(const qint32 i) {
    // cos(rad) is just sin(rad + pi/2) and we are at 4 fsc.
    return sin4fsc(i + 1);
}

// Public methods -----------------------------------------------------------------------------------------------------

Comb::Comb()
    : configurationSet(false)
{
}

qint32 Comb::Configuration::getLookBehind() const {
    if (dimensions == 3) {
        // In 3D mode, we need to see the previous frame
        return 1;
    }

    return 0;
}

qint32 Comb::Configuration::getLookAhead() const {
    if (dimensions == 3) {
        // ... and also the next frame
        return 2;
    }

    return 0;
}

// Return the current configuration
const Comb::Configuration &Comb::getConfiguration() const {
    return configuration;
}

// Set the comb filter configuration parameters
void Comb::updateConfiguration(const LdDecodeMetaData::VideoParameters &_videoParameters, const Comb::Configuration &_configuration)
{
    // Copy the configuration parameters
    videoParameters = _videoParameters;
    configuration = _configuration;

    // Range check the frame dimensions
    if (videoParameters.fieldWidth > MAX_WIDTH) qCritical() << "Comb::Comb(): Frame width exceeds allowed maximum!";
    if (((videoParameters.fieldHeight * 2) - 1) > MAX_HEIGHT) qCritical() << "Comb::Comb(): Frame height exceeds allowed maximum!";

    // Range check the video start
    if (videoParameters.activeVideoStart < 16) qCritical() << "Comb::Comb(): activeVideoStart must be > 16! (currently " << videoParameters.activeVideoStart << ")";

    // Check the sample rate is close to 4 * fSC.
    // Older versions of ld-decode used integer approximations, so this needs
    // to be an approximate comparison.
    if (fabs((videoParameters.sampleRate / videoParameters.fSC) - 4.0) > 1.0e-6)
    {
        qCritical() << "Data is not in 4fsc sample rate, color decoding will not work properly!";
    }

    configurationSet = true;
}

// -----------------------------------------------------------------------------
// [REVISED] decodeFrames: 4-Field Block / 2-Field Step (Overlap-Add)
// -----------------------------------------------------------------------------
void Comb::decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                        QVector<ComponentFrame> &componentFrames)
{
    assert(configurationSet);
    assert((componentFrames.size() * 2) == (endIndex - startIndex));

    // Cache to hold FrameBuffers because OLA requires adding to future frames
    // Key: Frame Index (relative to input array)
    QMap<int, std::shared_ptr<FrameBuffer>> bufferCache;

    // Helper: Get existing buffer or create new one populated with fields
    auto getFrameBuffer = [&](int frameIdx) -> std::shared_ptr<FrameBuffer> {
        if (bufferCache.contains(frameIdx)) {
            return bufferCache[frameIdx];
        }

        auto buf = std::make_shared<FrameBuffer>(videoParameters, configuration);
        
        // Calculate absolute field indices in inputFields
        int fieldIdx1 = startIndex + frameIdx * 2;
        int fieldIdx2 = fieldIdx1 + 1;

        if (fieldIdx1 >= 0 && fieldIdx2 < inputFields.size()) {
            buf->loadFields(inputFields[fieldIdx1], inputFields[fieldIdx2]);
            // Pre-calculate 1D/2D for fallback
            buf->split1D();
            buf->split2D();
        } 
        // Else: buffer remains black (boundary handling)

        bufferCache.insert(frameIdx, buf);
        return buf;
    };

    // Step by 2 fields (1 frame)
    // 4 Fields Block Logic: [Current, Next]
    for (qint32 fieldIndex = startIndex; fieldIndex < endIndex; fieldIndex += 2) {
        int currentFrameIdx = (fieldIndex - startIndex) / 2;
        
        // Block = [Current, Next] (4 Fields)
        auto bufCurr = getFrameBuffer(currentFrameIdx);
        auto bufNext = getFrameBuffer(currentFrameIdx + 1);
        
        if (configuration.dimensions == 3) {
            // Process 4-field block
            // Result is accumulated into bufCurr AND bufNext
            bufCurr->split3D(*bufNext, currentFrameIdx);
        }
        
        // Output Current Frame
        if (currentFrameIdx >= 0 && currentFrameIdx < componentFrames.size()) {
            auto buf = bufCurr;
            
            componentFrames[currentFrameIdx].init(videoParameters);
            buf->setComponentFrame(componentFrames[currentFrameIdx]);

            if (configuration.dimensions == 3) {
                buf->finalizeOLA();
            }

            if (configuration.phaseCompensation) {
                buf->splitIQlocked();
            } else {
                buf->splitIQ();
                buf->adjustY();
            }
        
            buf->filterIQ();

            // Apply noise reduction
            buf->doCNR();
            buf->doYNR();

            // Transform I/Q to U/V
            buf->transformIQ(configuration.chromaGain, configuration.chromaPhase);

            // Frame is done, remove from cache
            bufferCache.remove(currentFrameIdx);
        }
    }
}

// Private methods ----------------------------------------------------------------------------------------------------

Comb::FrameBuffer::FrameBuffer(const LdDecodeMetaData::VideoParameters &videoParameters_,
                               const Configuration &configuration_)
    : videoParameters(videoParameters_), configuration(configuration_)
{
    frameHeight = ((videoParameters.fieldHeight * 2) - 1);
    irescale = (videoParameters.white16bIre - videoParameters.black16bIre) / 100;

    // Initialize Accumulators
    int safeWidth = videoParameters.fieldWidth;
    int safeHeight = videoParameters.fieldHeight * 2;
    accChroma.resize(safeHeight, std::vector<double>(safeWidth, 0.0));
    weightSum.resize(safeHeight, std::vector<double>(safeWidth, 0.0));

    // Initialize rawbuffer to black to avoid uninitialized reads
    int totalSamples = videoParameters.fieldWidth * frameHeight;
    rawbuffer.fill(0, totalSamples); 
}

inline qint32 Comb::FrameBuffer::getFieldID(qint32 lineNumber) const
{
    bool isFirstField = ((lineNumber % 2) == 0);

    return isFirstField ? firstFieldPhaseID : secondFieldPhaseID;
}

// NOTE:  lineNumber is presumed to be starting at 1.  (This lines up with how splitIQ calls it)
inline bool Comb::FrameBuffer::getLinePhase(qint32 lineNumber) const
{
    qint32 fieldID = getFieldID(lineNumber);
    bool isPositivePhaseOnEvenLines = (fieldID == 1) || (fieldID == 4);

    int fieldLine = (lineNumber / 2);
    bool isEvenLine = (fieldLine % 2) == 0;

    return isEvenLine ? isPositivePhaseOnEvenLines : !isPositivePhaseOnEvenLines;
}

// Interlace two source fields into the framebuffer.
void Comb::FrameBuffer::loadFields(const SourceField &firstField, const SourceField &secondField)
{
    // Interlace the input fields and place in the frame buffer
    qint32 fieldLine = 0;
    rawbuffer.clear();
    for (qint32 frameLine = 0; frameLine < frameHeight; frameLine += 2) {
        rawbuffer.append(firstField.data.mid(fieldLine * videoParameters.fieldWidth, videoParameters.fieldWidth));
        rawbuffer.append(secondField.data.mid(fieldLine * videoParameters.fieldWidth, videoParameters.fieldWidth));
        fieldLine++;
    }

    // Set the phase IDs for the frame
    firstFieldPhaseID = firstField.field.fieldPhaseID;
    secondFieldPhaseID = secondField.field.fieldPhaseID;

    // Clear clpbuffer
    for (qint32 buf = 0; buf < 3; buf++) {
        for (qint32 y = 0; y < MAX_HEIGHT; y++) {
            for (qint32 x = 0; x < MAX_WIDTH; x++) {
                clpbuffer[buf].pixel[y][x] = 0.0;
            }
        }
    }

    // No component frame yet
    componentFrame = nullptr;
}

// Extract chroma into clpbuffer[0] using a 1D bandpass filter.
//
// The filter is [-0.25, 0, 0.5, 0, -0.25], a gentle bandpass centred on fSC.
// So the output will contain all of the chroma signal, but also whatever luma
// components ended up in the same frequency range.
//
// This also acts as an alias removal pre-filter for the quadrature detector in
// splitIQ, so we use its result for split2D rather than the raw signal.
void Comb::FrameBuffer::split1D()
{
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get a pointer to the line's data
        const quint16 *line = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double tc1 = (line[h] - ((line[h - 2] + line[h + 2]) / 2.0)) / 2.0;

            // Record the 1D C value
            clpbuffer[0].pixel[lineNumber][h] = tc1;
        }
    }
}

// Extract chroma into clpbuffer[1] using a 2D 3-line adaptive filter.
//
// Because the phase of the chroma signal changes by 180 degrees from line to
// line, subtracting two adjacent lines that contain the same information will
// give you just the chroma signal. But real images don't necessarily contain
// the same information on every line.
//
// The "3-line adaptive" part means that we look at both surrounding lines to
// estimate how similar they are to this one. We can then compute the 2D chroma
// value as a blend of the two differences, weighted by similarity.
void Comb::FrameBuffer::split2D()
{
    // Dummy black line
    static constexpr double blackLine[MAX_WIDTH] = {0};

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get pointers to the surrounding lines of 1D chroma.
        // If a line we need is outside the active area, use blackLine instead.
        const double *previousLine = blackLine;
        if (lineNumber - 2 >= videoParameters.firstActiveFrameLine) {
            previousLine = clpbuffer[0].pixel[lineNumber - 2];
        }
        const double *currentLine = clpbuffer[0].pixel[lineNumber];
        const double *nextLine = blackLine;
        if (lineNumber + 2 < videoParameters.lastActiveFrameLine) {
            nextLine = clpbuffer[0].pixel[lineNumber + 2];
        }

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double kp, kn;

            // Summing the differences of the *absolute* values of the 1D chroma samples
            // will give us a low value if the two lines are nearly in phase (strong Y)
            // or nearly 180 degrees out of phase (strong C) -- i.e. the two cases where
            // the 2D filter is probably usable. Also give a small bonus if
            // there's a large signal (we think).
            kp  = fabs(fabs(currentLine[h]) - fabs(previousLine[h]));
            kp += fabs(fabs(currentLine[h - 1]) - fabs(previousLine[h - 1]));
            kp -= (fabs(currentLine[h]) + fabs(previousLine[h - 1])) * .10;
            kn  = fabs(fabs(currentLine[h]) - fabs(nextLine[h]));
            kn += fabs(fabs(currentLine[h - 1]) - fabs(nextLine[h - 1]));
            kn -= (fabs(currentLine[h]) + fabs(nextLine[h - 1])) * .10;

            // Map the difference into a weighting 0-1.
            // 1 means in phase or unknown; 0 means out of phase (more than kRange difference).
            const double kRange = 45 * irescale;
            kp = qBound(0.0, 1 - (kp / kRange), 1.0);
            kn = qBound(0.0, 1 - (kn / kRange), 1.0);

            double sc = 1.0;

            if ((kn > 0) || (kp > 0)) {
                // At least one of the next/previous lines has a good phase relationship.

                // If one of them is much better than the other, only use that one
                if (kn > (3 * kp)) kp = 0;
                else if (kp > (3 * kn)) kn = 0;

                sc = (2.0 / (kn + kp));
                if (sc < 1.0) sc = 1.0;
            } else {
                // Neither line has a good phase relationship.

                // But are they similar to each other? If so, we can use both of them!
                if ((fabs(fabs(previousLine[h]) - fabs(nextLine[h])) - fabs((nextLine[h] + previousLine[h]) * .2)) <= 0) {
                    kn = kp = 1;
                }

                // Else kn = kp = 0, so we won't extract any chroma for this sample.
                // (Some NTSC decoders fall back to the 1D chroma in this situation.)
            }

            // Compute the weighted sum of differences, giving the 2D chroma value
            double tc1;
            tc1  = ((currentLine[h] - previousLine[h]) * kp * sc);
            tc1 += ((currentLine[h] - nextLine[h]) * kn * sc);
            tc1 /= 4;

            clpbuffer[1].pixel[lineNumber][h] = tc1;
        }
    }
}

#ifndef IDX3
#define IDX3(t, y, x, Nt, Ny, Nx) ((t)*(Ny)*(Nx) + (y)*(Nx) + (x))
#endif

// [FIX] 4-Field Split3D with STRICT Patent Logic (Symmetry & Freq Weight)
void Comb::FrameBuffer::split3D(FrameBuffer &nextFrame, int frameIdx)
{
    const int Nx = 16;
    const int Ny = 16;
    const int Nt = 4;
    
    // 50% Overlap (Step 8) is required for perfect reconstruction with Sine Window
    const int STEP_X = 8;
    const int STEP_Y = 8;
    const int SC_X = 4;

    // FFTW Setup
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nt * Ny * Nx);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nt * Ny * Nx);
    fftw_plan p_fwd = fftw_plan_dft_3d(Nt, Ny, Nx, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan p_inv = fftw_plan_dft_3d(Nt, Ny, Nx, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Windows: Sine Window (Standard for 50% overlap OLA)
    std::vector<double> winX(Nx), winY(Ny), winT(Nt);
    for(int i=0; i<Nx; ++i) winX[i] = sin(M_PI * (i + 0.5) / Nx);
    for(int i=0; i<Ny; ++i) winY[i] = sin(M_PI * (i + 0.5) / Ny);
    for(int i=0; i<Nt; ++i) winT[i] = sin(M_PI * (i + 0.5) / Nt); 

    FrameBuffer* frames[2] = { this, &nextFrame };

    // [CHANGE 1] Loop Range: Start BEFORE the image, End AFTER the image
    // Ensures edges are covered by the window center.
    int startY = videoParameters.firstActiveFrameLine - (Ny / 2); 
    int endY = videoParameters.lastActiveFrameLine; 
    
    int startX = videoParameters.activeVideoStart - (Nx / 2);
    int endX = videoParameters.activeVideoEnd;

    for (int y = startY; y < endY; y += STEP_Y) {
        for (int x = startX; x < endX; x += STEP_X) {

            // --- A. Fill Input (With Padding) ---
            for(int i=0; i < Nt*Ny*Nx; ++i) { in[i][0] = 0.0; in[i][1] = 0.0; }

            double blockDC = 0.0;
            int pixelCount = 0;

            for (int f = 0; f < 2; ++f) { 
                for (int sub_t = 0; sub_t < 2; ++sub_t) { 
                    int t = f * 2 + sub_t;
                    bool isOddField = (t % 2 != 0); 
                    
                    for (int dy = 0; dy < Ny; ++dy) {
                        int absY = y + dy;
                        // [CHECK] Is this line inside the video?
                        bool isYInside = (absY >= videoParameters.firstActiveFrameLine) && (absY < videoParameters.lastActiveFrameLine);
                        
                        // Interlace Check (Must match field polarity)
                        bool isOddLine = (absY % 2 != 0);
                        if (isOddLine != isOddField) continue; // Zero Pad for Interlace

                        // Boundary Padding Logic:
                        // If outside Y range, we leave it as 0.0 (Black Padding).
                        // If inside Y range, we check X range.
                        if (isYInside) {
                            const quint16 *lineData = frames[f]->rawbuffer.data() + (absY * videoParameters.fieldWidth);
                            for (int dx = 0; dx < Nx; ++dx) {
                                int absX = x + dx;
                                bool isXInside = (absX >= videoParameters.activeVideoStart) && (absX < videoParameters.activeVideoEnd);
                                
                                if (isXInside) {
                                    // Valid Pixel
                                    double val = (double)lineData[absX];
                                    int i = IDX3(t, dy, dx, Nt, Ny, Nx);
                                    in[i][0] = val; 
                                    blockDC += val;
                                    pixelCount++;
                                }
                                // Else: Outside X range -> Leave as 0.0 (Padding)
                            }
                        }
                        // Else: Outside Y range -> Leave as 0.0 (Padding)
                    }
                }
            }
            if (pixelCount > 0) blockDC /= (double)pixelCount;

            // DC Removal & Windowing
            for(int t=0; t<Nt; ++t) {
                bool isOddField = (t % 2 != 0);
                for(int dy=0; dy<Ny; ++dy) {
                    int absY = y + dy;
                    bool isYInside = (absY >= videoParameters.firstActiveFrameLine) && (absY < videoParameters.lastActiveFrameLine);
                    bool isOddLine = (absY % 2 != 0); // Polarity check based on absolute Y

                    for (int dx = 0; dx < Nx; ++dx) {
                        int absX = x + dx;
                        bool isXInside = (absX >= videoParameters.activeVideoStart) && (absX < videoParameters.activeVideoEnd);
                        
                        int idx = IDX3(t, dy, dx, Nt, Ny, Nx);

                        // Only remove DC if it was a real pixel
                        // Padded pixels (0.0) should NOT have DC subtracted (0 - DC = large jump)
                        // However, standard windowing applies to everything. 
                        // To keep edges smooth, we treat padding as "Black" (0.0).
                        
                        if (isYInside && isXInside && (isOddLine == isOddField)) {
                            in[idx][0] = (in[idx][0] - blockDC) * winT[t] * winY[dy] * winX[dx];
                        } else {
                            // Padding or Interlace Gap -> 0.0 * Window = 0.0
                            in[idx][0] = 0.0;
                        }
                    }
                }
            }

            // --- B. FFT ---
            fftw_execute(p_fwd);

            // =========================================================
            // [DATA GENERATION] Deterministic Subsampling & Thread-Safe Export
            // =========================================================
            /*
            //./ld-chroma-decoder -t 1 -f ntsc3d 必须使用单线程
            static bool dump_enabled = true; 
            // 记得跑第二次时改为 "train_target.bin"
            static const char* dump_filename = "/home/a/train_input.bin"; 
            
            static std::mutex dump_mutex; 

            // 确定性哈希采样 (Deterministic Hashing)
            // 不使用随机数，而是根据 (Frame, Y, X) 计算一个固定的 ID
            // 确保 Input 和 Target 两次运行时，选中的块是完全一一对应的。
            
            uint32_t seed = (uint32_t)frameIdx;
            seed = seed * 31 + (uint32_t)y;
            seed = seed * 31 + (uint32_t)x;
            // 简单的混淆算法 (Xorshift style)
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            
            // 采样率 5% (seed % 100 < 5)
            // 这样对于同一个位置的块，无论何时运行，结果都一样。
            bool should_save = (seed % 100) < 5; 

            if (dump_enabled && should_save) {
                std::lock_guard<std::mutex> lock(dump_mutex);
                
                static std::ofstream dumpFile;
                if (!dumpFile.is_open()) {
                    dumpFile.open(dump_filename, std::ios::binary | std::ios::out | std::ios::app);
                }

                if (dumpFile.is_open()) {
                    dumpFile.write(reinterpret_cast<const char*>(out), sizeof(fftw_complex) * Nt * Ny * Nx);
                }
            }
            // =========================================================
            */
            

            // --- C. Logic: Patent Compliant Frequency Dependent LUT (Full 3D) ---
            // =========================================================
            // [AI INFERENCE] Neural Network Logic (Linux Version)
            // =========================================================
            
            // 1. 初始化 ONNX Session (静态单例，只加载一次)
            static std::unique_ptr<Ort::Env> env;
            static std::unique_ptr<Ort::Session> session;
            static bool model_loaded = false;
            
            if (!model_loaded) {
                try {
                    // 初始化环境
                    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NTSC_AI");
                    
                    // Session 配置
                    Ort::SessionOptions session_options;
                    session_options.SetIntraOpNumThreads(11); // 单线程推理足够快了
                    
                    // 加载模型
                    // [Linux] 路径直接用字符串，不需要 L""
                    // 请确保 chroma_net.onnx 文件在运行目录下，或者写绝对路径
                    const char* model_path = "/home/ethan/git/vhs-decode/tools/chroma_net.onnx"; 
                    
                    session = std::make_unique<Ort::Session>(*env, model_path, session_options);
                    model_loaded = true;
                    qDebug() << "AI: ONNX Model loaded successfully from" << model_path;
                } catch (const std::exception& e) {
                    qCritical() << "AI: Failed to load ONNX model:" << e.what();
                    // 如果加载失败，程序可能需要退出或回退到传统逻辑
                }
            }

            if (model_loaded) {
                // 2. 准备输入 Tensor
                // Shape: [Batch=1, Channel=2, Depth=4, Height=16, Width=16]
                std::vector<int64_t> input_shape = {1, 2, 4, 16, 16};
                size_t input_element_count = 2048; // 1*2*4*16*16
                std::vector<float> input_tensor_values(input_element_count);
                
                // 填充数据：必须严格遵守 Python 训练时的顺序 [Mag, RefMag]
                int ptr = 0;
                
                // Channel 0: Original Magnitude
                for (int t = 0; t < Nt; ++t) {
                    for (int y = 0; y < Ny; ++y) {
                        for (int x = 0; x < Nx; ++x) {
                            int idx = IDX3(t, y, x, Nt, Ny, Nx);
                            double mag = sqrt(out[idx][0]*out[idx][0] + out[idx][1]*out[idx][1]);
                            input_tensor_values[ptr++] = (float)mag;
                        }
                    }
                }
                
                // Channel 1: Reflected Magnitude
                for (int t = 0; t < Nt; ++t) {
                    // Ref T: (2 - t) % 4
                    int ref_t = (2 - t) % 4;
                    if (ref_t < 0) ref_t += 4;

                    for (int y = 0; y < Ny; ++y) {
                        // Ref Y: (16 - x) % 16
                        int ref_y = (16 - y) % 16;
                        for (int x = 0; x < Nx; ++x) {
                            // Ref X: (8 - x) % 16
                            int ref_x = (8 - x) % 16;
                            if (ref_x < 0) ref_x += 16;
                            
                            int idx_ref = IDX3(ref_t, ref_y, ref_x, Nt, Ny, Nx);
                            double mag_ref = sqrt(out[idx_ref][0]*out[idx_ref][0] + out[idx_ref][1]*out[idx_ref][1]);
                            
                            input_tensor_values[ptr++] = (float)mag_ref;
                        }
                    }
                }

                // 3. 创建 Tensor
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_tensor_values.data(), input_element_count, input_shape.data(), input_shape.size()
                );

                // 4. 运行推理
                const char* input_names[] = {"input"};
                const char* output_names[] = {"output"};
                
                auto output_tensors = session->Run(
                    Ort::RunOptions{nullptr}, 
                    input_names, &input_tensor, 1, 
                    output_names, 1
                );

                // 5. 获取输出并应用 Mask
                // Output Shape: [1, 1, 4, 16, 16]
                float* mask_data = output_tensors[0].GetTensorMutableData<float>();
                
                int mask_idx = 0;
                for (int t = 0; t < Nt; ++t) {
                    for (int y = 0; y < Ny; ++y) {
                        for (int x = 0; x < Nx; ++x) {
                            int idx = IDX3(t, y, x, Nt, Ny, Nx);
                            float gain = mask_data[mask_idx++];
                            
                            // 应用神经网络算出来的增益！
                            out[idx][0] *= gain;
                            out[idx][1] *= gain;
                        }
                    }
                }
            }
            // =========================================================   
                 
            // --- D. IFFT ---
            fftw_execute(p_inv);

            // --- E. Accumulate (With Boundary Guards) ---
            for (int t = 0; t < Nt; ++t) {
                int f_idx = t / 2;
                bool isOddField = (t % 2 != 0);
                FrameBuffer* targetFrame = frames[f_idx];

                for (int dy = 0; dy < Ny; ++dy) {
                    int absY = y + dy;
                    
                    // [CHECK] Only accumulate if valid Y
                    if (absY < videoParameters.firstActiveFrameLine || absY >= videoParameters.lastActiveFrameLine) continue;
                    
                    // Interlace check
                    if ((absY % 2) != isOddField) continue;

                    for (int dx = 0; dx < Nx; ++dx) {
                        int absX = x + dx;
                        
                        // [CHECK] Only accumulate if valid X
                        if (absX < videoParameters.activeVideoStart || absX >= videoParameters.activeVideoEnd) continue;

                        int idx = IDX3(t, dy, dx, Nt, Ny, Nx);
                        
                        double val = in[idx][0] / (double)(Nt * Ny * Nx);
                        double w = winT[t] * winY[dy] * winX[dx];
                        
                        targetFrame->accChroma[absY][absX] += val * w;
                        targetFrame->weightSum[absY][absX] += w * w;
                    }
                }
            }
        }
    }
    fftw_destroy_plan(p_fwd); fftw_destroy_plan(p_inv);
    fftw_free(in); fftw_free(out);
}

// [MODIFIED] Finalize OLA: Removed 2D Fallback
void Comb::FrameBuffer::finalizeOLA() {
    int writeHeight = videoParameters.lastActiveFrameLine;
    int writeWidth = videoParameters.activeVideoEnd;
    
    for (int y = videoParameters.firstActiveFrameLine; y < writeHeight; ++y) {
        for (int x = videoParameters.activeVideoStart; x < writeWidth; ++x) {
            double w = weightSum[y][x];
            
            // Because we now cover edges with padding, weightSum should > 0 everywhere.
            // If for some reason it's 0 (e.g. extremely corner), result is 0.
            if (w > 0.00001) {
                clpbuffer[2].pixel[y][x] = accChroma[y][x] / w;
            } else {
                clpbuffer[2].pixel[y][x] = 0.0; // No 2D fallback, just black chroma
            }
        }
    }
}

// ... (Rest of auxiliary functions) ...
void Comb::FrameBuffer::getBestCandidate(qint32 lineNumber, qint32 h, const FrameBuffer &previousFrame, const FrameBuffer &nextFrame, qint32 &bestIndex, double &bestSample) const {
    Candidate candidates[8];
    static constexpr double LINE_BONUS = -2.0;
    static constexpr double FIELD_BONUS = LINE_BONUS - 2.0;
    static constexpr double FRAME_BONUS = FIELD_BONUS - 2.0;

    // 1D: Same line, 2 samples left and right
    candidates[CAND_LEFT]  = getCandidate(lineNumber, h, *this, lineNumber, h - 2, 0);
    candidates[CAND_RIGHT] = getCandidate(lineNumber, h, *this, lineNumber, h + 2, 0);

    // 2D: Same field, 1 line up and down
    candidates[CAND_UP]   = getCandidate(lineNumber, h, *this, lineNumber - 2, h, LINE_BONUS);
    candidates[CAND_DOWN] = getCandidate(lineNumber, h, *this, lineNumber + 2, h, LINE_BONUS);

    // Immediately adjacent lines in previous/next field
    if (getLinePhase(lineNumber) == getLinePhase(lineNumber - 1)) {
        candidates[CAND_PREV_FIELD] = getCandidate(lineNumber, h, previousFrame, lineNumber - 1, h, FIELD_BONUS);
        candidates[CAND_NEXT_FIELD] = getCandidate(lineNumber, h, *this, lineNumber + 1, h, FIELD_BONUS);
    } else {
        candidates[CAND_PREV_FIELD] = getCandidate(lineNumber, h, *this, lineNumber - 1, h, FIELD_BONUS);
        candidates[CAND_NEXT_FIELD] = getCandidate(lineNumber, h, nextFrame, lineNumber + 1, h, FIELD_BONUS);
    }

    // Previous/next frame, same position
    candidates[CAND_PREV_FRAME] = getCandidate(lineNumber, h, previousFrame, lineNumber, h, FRAME_BONUS);
    candidates[CAND_NEXT_FRAME] = getCandidate(lineNumber, h, nextFrame, lineNumber, h, FRAME_BONUS);

    if (configuration.adaptive) {
        // Find the candidate with the lowest penalty
        bestIndex = 0;
        for (qint32 i = 1; i < NUM_CANDIDATES; i++) {
            if (candidates[i].penalty < candidates[bestIndex].penalty) bestIndex = i;
        }
    } else {
        // Adaptive mode is disabled - do 3D against the previous frame
        bestIndex = CAND_PREV_FRAME;
    }

    bestSample = candidates[bestIndex].sample;
}

// Evaluate a candidate for 3D decoding
Comb::FrameBuffer::Candidate Comb::FrameBuffer::getCandidate(qint32 refLineNumber, qint32 refH,
                                                             const FrameBuffer &frameBuffer, qint32 lineNumber, qint32 h,
                                                             double adjustPenalty) const
{
    Candidate result;
    result.sample = frameBuffer.clpbuffer[0].pixel[lineNumber][h];

    // If the candidate is outside the active region (vertically), it's not viable
    if (lineNumber < videoParameters.firstActiveFrameLine || lineNumber >= videoParameters.lastActiveFrameLine) {
        result.penalty = 1000.0;
        return result;
    }

    // The target sample should have 180 degrees phase difference from the reference.
    // If it doesn't (e.g. because it's a blank frame or the player skipped), it's not viable.
    const qint32 wantPhase = (2 + (getLinePhase(refLineNumber) ? 2 : 0) + refH) % 4;
    const qint32 havePhase = ((frameBuffer.getLinePhase(lineNumber) ? 2 : 0) + h) % 4;
    if (wantPhase != havePhase) {
        result.penalty = 1000.0;
        return result;
    }

    // Pointers to the baseband data
    const quint16 *refLine = rawbuffer.data() + (refLineNumber * videoParameters.fieldWidth);
    const quint16 *candidateLine = frameBuffer.rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);

    // Penalty based on mean luma difference in IRE over surrounding three samples
    double yPenalty = 0.0;
    for (qint32 offset = -1; offset < 2; offset++) {
        const double refC = clpbuffer[1].pixel[refLineNumber][refH + offset];
        const double refY = refLine[refH + offset] - refC;

        const double candidateC = frameBuffer.clpbuffer[1].pixel[lineNumber][h + offset];
        const double candidateY = candidateLine[h + offset] - candidateC;

        yPenalty += fabs(refY - candidateY);
    }
    yPenalty = yPenalty / 3 / irescale;

    // Penalty based on mean I/Q difference in IRE over surrounding three samples
    double iqPenalty = 0.0;
    for (qint32 offset = -1; offset < 2; offset++) {
        // The reference and candidate are 180 degrees out of phase here, so negate one
        const double refC = clpbuffer[1].pixel[refLineNumber][refH + offset];
        const double candidateC = -frameBuffer.clpbuffer[1].pixel[lineNumber][h + offset];

        // I and Q samples alternate, so weight the two channels equally
        static constexpr double weights[] = {0.5, 1.0, 0.5};
        iqPenalty += fabs(refC - candidateC) * weights[offset + 1];
    }
    // Weaken this relative to luma, to avoid spurious colour in the 2D result from showing through
    iqPenalty = (iqPenalty / 2 / irescale) * 0.28;

    result.penalty = yPenalty + iqPenalty + adjustPenalty * configuration.chromaWeight;
    return result;
}

namespace {
    // Information about a line we're decoding.
    struct BurstInfo {
        double bsin, bcos;
    };

    // Rotate the burst angle to get the correct values.
    // We do the 33 degree rotation here to avoid computing it for every pixel.
    constexpr double ROTATE_SIN = 0.5446390350150271;
    constexpr double ROTATE_COS = 0.838670567945424;

    BurstInfo detectBurst(const quint16* lineData,
                          const LdDecodeMetaData::VideoParameters& videoParameters)
    {
        double bsin = 0, bcos = 0;

        // Find absolute burst phase relative to the reference carrier by
        // product detection.
        // For now we just use the burst on the current line, but we could possibly do some averaging with
        // neighbouring lines later if needed.
        for (qint32 i = videoParameters.colourBurstStart; i < videoParameters.colourBurstEnd; i++) {
            bsin += lineData[i] * sin4fsc(i);
            bcos += lineData[i] * cos4fsc(i);
        }

        // Normalise the sums above
        const qint32 colourBurstLength = videoParameters.colourBurstEnd - videoParameters.colourBurstStart;
        bsin /= colourBurstLength;
        bcos /= colourBurstLength;

        const double burstNorm = qMax(sqrt(bsin * bsin + bcos * bcos), 130000.0 / 128);

        bsin /= burstNorm;
        bcos /= burstNorm;

        const BurstInfo info{bsin, bcos};
        return info;
    }
}

// Split I and Q, taking burst phase into account.
void Comb::FrameBuffer::splitIQlocked()
{
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get a pointer to the line's data
        const quint16 *line = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);
        // Calculate burst phase
        const auto info = detectBurst(line, videoParameters);

        double *Y = componentFrame->y(lineNumber);
        double *I = componentFrame->u(lineNumber);
        double *Q = componentFrame->v(lineNumber);

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            const auto val = clpbuffer[configuration.dimensions - 1].pixel[lineNumber][h];

            // Demodulate the sine and cosine components.
            const auto lsin = val * sin4fsc(h) * 2;
            const auto lcos = val * cos4fsc(h) * 2;
            // Rotate the demodulated vector by the burst phase.
            const auto ti = (lsin * info.bcos - lcos * info.bsin);
            const auto tq = (lsin * info.bsin + lcos * info.bcos);

            // Invert Q and rotate to get the correct I/Q vector.
            // TODO: Needed to shift the chroma 1 sample to the right to get it to line up
            // may not get the first pixel in each line correct because of this.
            I[h + 1] = ti * ROTATE_COS - tq * -ROTATE_SIN;
            Q[h + 1] = -(ti * -ROTATE_SIN + tq * ROTATE_COS);
            // Subtract the split chroma part from the luma signal.
            Y[h] = line[h] - val;
        }
    }
}

// Spilt the I and Q
void Comb::FrameBuffer::splitIQ()
{
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get a pointer to the line's data
        const quint16 *line = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);

        double *Y = componentFrame->y(lineNumber);
        double *I = componentFrame->u(lineNumber);
        double *Q = componentFrame->v(lineNumber);

        bool linePhase = getLinePhase(lineNumber);

        double si = 0, sq = 0;
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            qint32 phase = h % 4;

            double cavg = clpbuffer[configuration.dimensions - 1].pixel[lineNumber][h];

            if (linePhase) cavg = -cavg;

            switch (phase) {
                case 0: sq = cavg; break;
                case 1: si = -cavg; break;
                case 2: sq = -cavg; break;
                case 3: si = cavg; break;
                default: break;
            }

            Y[h] = line[h];
            I[h] = si;
            Q[h] = sq;
        }
    }
}

// Filter the IQ from the component frame
void Comb::FrameBuffer::filterIQ()
{
    auto iqFilter = makeFIRFilter(c_colorlp_b);

    // Temporary output buffer for the filter
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
    std::vector<double> tempBuf(width);

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *I = componentFrame->u(lineNumber) + videoParameters.activeVideoStart;
        double *Q = componentFrame->v(lineNumber) + videoParameters.activeVideoStart;

        // Apply filter to I
        iqFilter.apply(I, tempBuf.data(), width);
        std::copy(tempBuf.begin(), tempBuf.end(), I);

        // Apply filter to Q
        iqFilter.apply(Q, tempBuf.data(), width);
        std::copy(tempBuf.begin(), tempBuf.end(), Q);
    }
}

// Remove the colour data from the baseband (Y)
void Comb::FrameBuffer::adjustY()
{
    // remove color data from baseband (Y)
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *Y = componentFrame->y(lineNumber);
        double *I = componentFrame->u(lineNumber);
        double *Q = componentFrame->v(lineNumber);

        bool linePhase = getLinePhase(lineNumber);

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double comp = 0;
            qint32 phase = h % 4;

            switch (phase) {
                case 0: comp = -Q[h]; break;
                case 1: comp = I[h]; break;
                case 2: comp = Q[h]; break;
                case 3: comp = -I[h]; break;
                default: break;
            }

            if (!linePhase) comp = -comp;
            Y[h] -= comp;
        }
    }
}

/*
 * This applies an FIR coring filter to both I and Q color channels.  It's a simple (crude?) NR technique used
 * by LD players, but effective especially on the Y/luma channel.
 *
 * A coring filter removes high frequency components (.4mhz chroma, 2.8mhz luma) of a signal up to a certain point,
 * which removes small high frequency noise.
 */

void Comb::FrameBuffer::doCNR()
{
    if (configuration.cNRLevel == 0) return;

    // nr_c is the coring level
    const double nr_c = configuration.cNRLevel * irescale;

    // High-pass filters for I/Q
    auto iFilter(f_nrc);
    auto qFilter(f_nrc);

    // Filter delay (since it's a symmetric FIR filter)
    const qint32 delay = c_nrc_b.size() / 2;

    // High-pass result
    // TODO: Cache arrays instead of reallocating every field.
    std::vector<double> hpI(videoParameters.activeVideoEnd + delay);
    std::vector<double> hpQ(videoParameters.activeVideoEnd + delay);


    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *I = componentFrame->u(lineNumber);
        double *Q = componentFrame->v(lineNumber);

        // Feed zeros into the filter outside the active area
        for (qint32 h = videoParameters.activeVideoStart - delay; h < videoParameters.activeVideoStart; h++) {
            iFilter.feed(0.0);
            qFilter.feed(0.0);
        }
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            hpI[h] = iFilter.feed(I[h]);
            hpQ[h] = qFilter.feed(Q[h]);
        }
        for (qint32 h = videoParameters.activeVideoEnd; h < videoParameters.activeVideoEnd + delay; h++) {
            hpI[h] = iFilter.feed(0.0);
            hpQ[h] = qFilter.feed(0.0);
        }

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            // Offset to cover the filter delay
            double ai = hpI[h + delay];
            double aq = hpQ[h + delay];

            // Clip the filter strength
            if (fabs(ai) > nr_c) {
                ai = (ai > 0) ? nr_c : -nr_c;
            }
            if (fabs(aq) > nr_c) {
                aq = (aq > 0) ? nr_c : -nr_c;
            }

            I[h] -= ai;
            Q[h] -= aq;
        }
    }
}

void Comb::FrameBuffer::doYNR()
{
    if (configuration.yNRLevel == 0) return;

    // nr_y is the coring level
    double nr_y = configuration.yNRLevel * irescale;

    // High-pass filter for Y
    auto yFilter(f_nr);

    // Filter delay (since it's a symmetric FIR filter)
    const qint32 delay = c_nr_b.size() / 2;

    // High-pass result
    std::vector<double> hpY(videoParameters.activeVideoEnd + delay);

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *Y = componentFrame->y(lineNumber);

        // Feed zeros into the filter outside the active area
        for (qint32 h = videoParameters.activeVideoStart - delay; h < videoParameters.activeVideoStart; h++) {
            yFilter.feed(0.0);
        }
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            hpY[h] = yFilter.feed(Y[h]);
        }
        for (qint32 h = videoParameters.activeVideoEnd; h < videoParameters.activeVideoEnd + delay; h++) {
            hpY[h] = yFilter.feed(0.0);
        }

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            // Offset to cover the filter delay
            double a = hpY[h + delay];

            // Clip the filter strength
            if (fabs(a) > nr_y) {
                a = (a > 0) ? nr_y : -nr_y;
            }

            Y[h] -= a;
        }
    }
}

// Transform I/Q into U/V, and apply chroma gain
void Comb::FrameBuffer::transformIQ(double chromaGain, double chromaPhase)
{
    // Compute components for the rotation vector
    const double theta = ((33 + chromaPhase) * M_PI) / 180;
    const double bp = sin(theta) * chromaGain;
    const double bq = cos(theta) * chromaGain;

    // Apply the vector to all the samples
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *I = componentFrame->u(lineNumber);
        double *Q = componentFrame->v(lineNumber);

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double U = (-bp * I[h]) + (bq * Q[h]);
            double V = ( bq * I[h]) + (bp * Q[h]);

            I[h] = U;
            Q[h] = V;
        }
    }
}

// Overlay the 3D filter map onto the output
void Comb::FrameBuffer::overlayMap(const FrameBuffer &previousFrame, const FrameBuffer &nextFrame)
{
    qDebug() << "Comb::FrameBuffer::overlayMap(): Overlaying map onto output";

    // Create a canvas for colour conversion
    FrameCanvas canvas(*componentFrame, videoParameters);

    // Convert CANDIDATE_SHADES into Y'UV form
    FrameCanvas::Colour shades[NUM_CANDIDATES];
    for (qint32 i = 0; i < NUM_CANDIDATES; i++) {
        const quint32 shade = CANDIDATE_SHADES[i];
        shades[i] = canvas.rgb(
            ((shade >> 16) & 0xff) << 8,
            ((shade >> 8) & 0xff) << 8,
            (shade & 0xff) << 8
        );
    }

    // For each sample in the frame...
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *U = componentFrame->u(lineNumber);
        double *V = componentFrame->v(lineNumber);

        // Fill the output frame with the RGB values
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            // Select the best candidate
            qint32 bestIndex;
            double bestSample;
            getBestCandidate(lineNumber, h, previousFrame, nextFrame, bestIndex, bestSample);

            // Leave Y' the same, but replace UV with the appropriate shade
            U[h] = shades[bestIndex].u;
            V[h] = shades[bestIndex].v;
        }
    }
}
