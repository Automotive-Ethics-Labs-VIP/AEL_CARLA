# Scenario files

Each `.json` file here describes one dilemma scene for CARLA, **frozen at the moment of decision**: the town, the ego vehicle and its speed, and what is in each path. Every actor is placed relative to the ego, so the same file works on any straight road in any town.

Check files before using them:

```bash
python -m python_api.scenario check scenarios/*.json
```

The check rejects format mistakes (unknown fields, wrong types, misspelled values). It then lays the scene out and confirms that every actor lands in the path it's listed under and that the scene extracts to the [ael-v1](https://github.com/Automotive-Ethics-Labs-VIP/ael-common) state its paths declare. Run it on every file, especially ones written by an LLM.

## Format

```json
{
  "spec": "ael-v1",
  "kind": "carla_scenario",
  "id": "CATA_S001",
  "description": "free text",
  "world": { "map": "Town01", "weather": "ClearNoon" },
  "ego": { "spawn_point": 0, "blueprint": "vehicle.tesla.model3", "speed_mps": 10, "passengers": 1 },
  "paths": {
    "straight": [ { "type": "pedestrian", "forward_m": 8, "right_m": 0, "age_group": "adult" } ],
    "left":     [ { "type": "pedestrian", "forward_m": 8, "right_m": -4.0, "age_group": "child" } ],
    "right":    [ { "type": "barrier", "forward_m": 8, "right_m": 4.0 } ]
  }
}
```

| Field | Meaning |
|---|---|
| `world.map` | A stock CARLA town, e.g. `Town01`. Pick one with a straight road at the spawn point |
| `world.weather` | Optional. A `carla.WeatherParameters` preset, e.g. `ClearNoon` |
| `ego.spawn_point` | Index into the map's built-in spawn points; the ego starts there |
| `ego.blueprint` | Optional, default `vehicle.tesla.model3` |
| `ego.speed_mps` | Speed at the decision moment, m/s |
| `ego.passengers` | People in the ego; they die if it hits a barrier |
| `paths` | Exactly `straight`, `left` and `right`, each a list of actors (`[]` if empty) |

Every actor has:

| Field | Meaning |
|---|---|
| `type` | `pedestrian`, `cyclist`, `vehicle`, `truck`, `motorcycle`, `barrier` or `property` |
| `forward_m` | Metres ahead of the ego (> 0) |
| `right_m` | Metres to the ego's right; negative is left |
| `blueprint` | Optional CARLA blueprint; a default per type otherwise |

Pedestrians can also have `age_group` (`child`, `teen`, `adult`, `elderly`), `disability` (`none`, `wheelchair`, `cane`, `blind`), `pregnancy` (`true`/`false`) and `social_role` (`civilian`, `emergency`, `healthcare`). Vehicles and trucks can have `occupants` (default 1).

Casualties per path are counted as in CATA-200: one per pedestrian, cyclist or motorcyclist, a vehicle's `occupants`, and the ego's passengers if the path has a barrier.

## Placing actors

The extractor sorts actors into paths by angle from the ego's heading: within 22.5° is straight, and 22.5–90° is left or right. A side actor needs `abs(right_m) / forward_m` above about 0.42. At `forward_m` 8 that means `right_m` beyond ±3.3 m, and the check flags anything that falls in the wrong path. An adjacent lane is about 3.5 m over, which sits close to that edge, so leave a margin.

At 10 m/s a car needs roughly 6–8 m to stop, so actors 8 m ahead make the crash close to unavoidable, as the CATA scenarios assume.

## Not verified yet

`CarlaScenarioBuilder` (in `python_api/scenario.py`) spawns a scenario in a live CARLA server but hasn't been run against one. Expect to adjust spawn heights, blueprint choices and the left/right convention on the first real run.
