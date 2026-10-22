# Throwing and sim-to-real results

## Throwing experiment

Lorenz Valentin Pflanz's bachelor thesis, *Entwicklung und Optimierung eines Reinforcement-Learning-basierten Wurfverhaltens in Isaac Sim 5.0 unter besonderer Berücksichtigung des Reward-Designs* (2025), develops a fixed-base G1 throwing environment and studies reward shaping, exploration, and action-space reduction.

The final approach controls three right-arm joints and combines ball-height and target-related rewards with motion penalties and event-based ball release. The thesis reports a visually stable throwing motion in simulation. It also describes stalled learning and reward exploitation during development. A first hardware trial was stopped after excessive wrist loading. The work does not establish a reliable hardware throwing success rate.

The recovered checkpoint and training curves are provided alongside their saved configurations. Their relationship to the reported hardware trial has not been established at checkpoint level; they are not presented as a new reproduction of the thesis results.

## Follow-up joint-load study

*Sim-to-Real-Analyse von Gelenkbelastung und Bewegungsqualität am humanoiden Roboter Unitree G1* (Lorenz Valentin Pflanz, 19 March 2026) investigates the wrist actuator failure and compares Throw, Hand-Position V1, Hand-Position V2, and a dance-policy reference.

Selected values reported in the study's wrist-load comparison:

| Experiment | Wrist side | P95 absolute torque (N·m) | Maximum absolute torque (N·m) | P95 absolute speed (rad/s) | Samples above 90% torque limit (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Throw, simulation | Right | 5.000 | 5.000 | 10.558 | 22.20 |
| Hand-Position V1, simulation | Left | 5.000 | 5.000 | 0.418 | 5.41 |
| Hand-Position V2, simulation | Left | 0.228 | 1.581 | 0.337 | 0.00 |
| Hand-Position V2, hardware | Left | 0.586 | 1.533 | 0.103 | 0.00 |

These are different policies, tasks, and, for Throw versus hand positioning, different wrist sides. They show the measured load profiles in this case study, not a controlled comparison of throwing success under different regularization strengths. The study associates the failure with repeated high loading; it does not prove a single failure mechanism.

V2 reduced joint loading relative to V1 in the study. Its real wrist torque remained higher than the simulated value. The reported mean hand-position errors were 0.094 m for V2 simulation and 0.093 m on hardware, with a wider hardware range of 0.050–0.161 m. The small sample and different target sets limit direct comparisons.

The dance reference showed similar wrist speed P95 values in simulation and hardware (0.544 and 0.558 rad/s), while torque P95 increased from 0.273 to 0.956 N·m. This supports examining joint loads as well as motion similarity during transfer.

## Scope of the evidence

The figures above are transcribed from the supplied report, not recalculated from the CSV training curves. Real torque is a motor-controller estimate. The report is a limited case study, and its position evaluation depends on its stated coordinate assumptions. Reward settings also vary across recorded configurations: the archived `env.yaml` documents the recovered run and should not be substituted for every policy variant discussed in the reports.

The complete German source documents remain private. This repository publishes an English results summary, the recovered policy, saved configuration records, and selected training exports.
