# Repository guidelines

- Avoid functions or methods that only forward arguments to another function.
  Keep them only when they define a public compatibility layer or isolate an
  external library, and make that purpose clear.
- Keep function signatures small. Group related values in a configuration,
  state, action, or result class.
- Keep short function calls on one line when they remain readable.
- Prefer plain names and direct code over unnecessary jargon, comments, and
  abstractions.
