# Design decisions

- Kept the existing React/Vite stack and API contracts instead of adding a second UI framework;
  this keeps the product build offline-capable and avoids replacing working accessibility tests.
- Introduced one token file for colour, spacing-adjacent visual values, motion, and shadows.
- Reframed the answer as a conversation: verdict and reading copy come first, while evidence stays
  behind numbered citation chips and the existing source drawer.
- Used a custom scope chip rather than a native select to make the composer consistent across
  desktop and mobile.
- Preserved the existing document, evaluation, history, and settings routes while moving navigation
  into a responsive sidebar.
