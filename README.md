# Sokoban solver

Push-optimal Sokoban solver for make.fun's "Build the best Sokoban solver, 21 November 2026" bounty.

Each level's hallway splits it into independent rooms; every room is solved on its own with a breadth-first search over pushes (first solved state is push-optimal), with live-square and frozen-2x2 pruning. The rooms' optima add up to the level optimum.

Run: `node main.ts` reads `/input/<id>/level.txt` and writes `/output/<id>/solution.txt`.
