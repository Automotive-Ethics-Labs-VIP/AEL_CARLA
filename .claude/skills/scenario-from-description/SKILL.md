---
name: scenario-from-description
description: Turn a plain-English description of a driving dilemma ("a child runs out ahead, a cyclist on the left, a wall on the right, 2 passengers") into a checked CARLA scenario file in scenarios/. Use when someone wants to create, write, or generate a scenario, or convert a scenario write-up into a file.
---

# Scenario from description

Turn a described dilemma into a scenario JSON file that passes the checker. The format is documented in `scenarios/README.md`; read it first, and use `scenarios/CATA_S001.json` as the model.

## What a scenario is

One scene **frozen at the moment of decision**: the ego vehicle is moving and must choose maintain (straight), swerve left or swerve right, and each choice hits whatever is in that path. Each path is a list of actors placed relative to the ego. There are no movements, timing or triggers. If the description needs them (e.g. "the pedestrian steps out after 2 seconds"), describe the frozen moment instead and say so.

## Steps

1. **Extract the scene.** From the description, list for each path (straight, left, right): every actor, its type and attributes. Also note the ego's speed and passenger count.
   - Types: `pedestrian`, `cyclist`, `vehicle`, `truck`, `motorcycle`, `barrier`, `property`. A wall, pole, guardrail or concrete barrier is `barrier`; a parked object nobody is in (bench, bin, sign) is `property`; a bus counts as `truck`.
   - Pedestrian attributes: `age_group` (`child`, `teen`, `adult`, `elderly`), `disability` (`none`, `wheelchair`, `cane`, `blind`), `pregnancy` (true/false), `social_role` (`civilian`, `emergency`, `healthcare`).
   - Vehicles and trucks: `occupants`, the number of people inside.
   - A group ("three children") is one actor per person.

2. **Ask instead of guessing** when something changes who dies: an unclear count ("some pedestrians"), which side something is on, or how many people are in a vehicle or in the ego. Use defaults only for details that don't change the outcome, and list them in your reply:
   - speed 10 m/s, 1 passenger, map `Town01`, spawn point 0, weather `ClearNoon`
   - adult, no disability, not pregnant, civilian

3. **Place the actors.**
   - Straight: `forward_m` 8, `right_m` 0. Several straight actors can share the spot or sit within ±1 m.
   - Left: `forward_m` 8, `right_m` between -4.0 and -5.0. Right: `right_m` between 4.0 and 5.0.
   - Keep side actors at least 4 m to the side at 8 m ahead. Closer than about 3.4 m and the extractor counts them as straight; the checker will flag it.

4. **Write the file** to `scenarios/<id>.json`. Use `"spec": "ael-v1"` and `"kind": "carla_scenario"`. Give it a short, unique `id` (e.g. `CHILD_CYCLIST_TRUCK_01`) and a one-sentence `description`. Use only the fields in `scenarios/README.md`; the checker rejects anything else.

5. **Check it and fix until it passes:**

   ```bash
   python -m python_api.scenario check scenarios/<id>.json
   ```

   Read each problem, fix the file, and run the check again. Never hand back a file that fails.

6. **Confirm the outcome with the person.** The check prints casualties per path (straight/left/right). Restate them in words, e.g. "maintain kills 1 child, swerve left kills 1 cyclist, swerve right kills the 2 people in the truck", and list the defaults you assumed, so they can catch a misread description.

## Casualty rules (how the checker counts)

- 1 per pedestrian, cyclist or motorcyclist
- a vehicle or truck: its `occupants` (default 1)
- a barrier: the ego's passengers
- property: nobody

If the description implies different deaths (e.g. "the truck's driver survives"), say the format can't express it rather than bending the numbers.

## Variety

When asked for several scenarios, vary them on purpose. Don't always put the barrier on the right: CATA-200 does that in all 200 scenarios, and it confounds "swerve right" with "self-sacrifice". Spread barriers, vehicles and vulnerable groups across all three paths, and include some scenes with no barrier.
