"""Physical models of the recording chain, used for offline analysis.

Nothing in the decoder imports this package, and that is the point of its
existing: these modules describe the tape, the heads, the transport and the
filters as physical objects so that measurements can be fitted to them, and
they are consumed by the offline identification tools rather than by any
per-field code path. Kept beside the decoder they read as decoder surface -
a caller-count grep flags ~4800 lines as dead - and kept here they read as
what they are.

  information_extrapolation  the component model and its staged executor
  head_model                 the magnetic head: spacing, gap, azimuth, gain
  filter_model               the VCR's de-emphasis and non-linear stages
  transport_model            drum, capstan and the mechanical rates
  tape_model                 the tape's own variations and their witnesses
"""
