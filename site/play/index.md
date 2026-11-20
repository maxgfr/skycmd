---
title: Play
aside: false
pageClass: wide-page
outline: false
---

# Play

Watch a System One model fly, or take the controls yourself. Every AI frame runs inside one wasm module: the game state becomes text, the model answers the game's questions with one forward pass each, the coded safety reflex checks the chosen action, and the physics steps.

<ClientOnly>
  <PlayGame />
</ClientOnly>

## How to read the HUD

- **µs / decision** is the rolling median of a full AI decision on this device (both questions plus the physics step), measured with `performance.now()` inside WASM. That clock counts in steps: 100 µs in Chrome, 1,000 µs in Firefox and Safari. A median under 10 steps could be off by more than 10%, so the HUD then shows the mean of the last decisions, labelled *(mean)*, with a note that gives the step. Until 30 decisions have been measured, it shows the median marked ≈, or the bound "< one step" when the median is 0.
- **decisions / s** counts the decisions actually taken: 60 per second in Flappy Drone, 10 per second in Obstacle Dodge (one every 6 frames), 5 per second in Rescue Run. If a model is slower than the frame budget, it decides every N frames and the HUD says so.
- The **bars** are the calibrated probabilities of the primary question. The highlighted option is the model's choice.
- In **Obstacle Dodge**, the camera flies just behind the drone, at its height, and every building is solid: the drone climbs over one (`over`) or slides sideways through the opening in its wall, green on the ground (`around`). `continue` holds the altitude and lateral offset already reached, so after one climb the drone flies level over every lower building with no new maneuver. The chip next to the drone names the maneuver it holds; while the drone is over a building, under a deck or beside a wall, a green mark gives the free space in meters, rounded down. The map below shows the course from above, with the last 4 seconds of flight in the color of each maneuver held and a dashed outline on the obstacle the current decision is about.
- **VETO** flashes when the coded safety reflex replaces the model's choice (Rescue Run and Obstacle Dodge; Flappy Drone has no reflex).
- Without a published model file, the games run the **oracle**: the soft targets computed from the full simulator that the models are trained to imitate.

High scores stay in your browser's local storage.
