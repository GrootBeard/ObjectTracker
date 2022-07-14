# Things to do
- Improve logger
  - Add logging of considered (feasible) measurements for each track at each epoch
  - Add coloring based on if measuremnt is actual, clutter or originating from a diffrent track
- Create CLI application
  - Allows to display information per track
  - Allows to display information at each epoch
- Add abstraction for model
  - Abstract model should generate matrices H, F, Q etc. based on a set of parameters
  - Abstract model should also specify which values can be logged

