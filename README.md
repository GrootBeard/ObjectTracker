## Description
Use joint probabilistic data association algorithms to track multiple obects.

## Things to do
- Improve logger
  - Add logging of considered (feasible) measurements for each track at each epoch
  - Add coloring based on if measuremnt is actual, clutter or originating from a diffrent track
- Create CLI application
  - Allows to display information per track
  - Allows to display information at each epoch
  - Can generate ground-truth data and save to file
    - Specify number of objects, shapes of tracks, noise, clutter etc.
  - Load ground-truth data from file
- Add abstraction for model
  - Abstract model should generate matrices H, F, Q etc. based on a set of parameters
  - Abstract model should also specify which values can be logged
- Path creation and deletion
  - Several methods to perform this
