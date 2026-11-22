// Room-by-room push-optimal Sokoban solver.
//
// Idea: in every level the generator draws, the player starts in a
// one-square-wide hallway and the boxes live in rooms off it, each room
// opening onto the hallway through a single door. A box pushed out of a
// door into the hallway can never be pushed back, so no optimal solution
// ever moves a box between rooms: each room is an independent puzzle and
// the level optimum is the sum of the rooms' optima.
//
// So the solver splits a level into rooms, solves each room on its own
// with a breadth-first search over pushes (the first solved state a BFS
// meets is push-optimal), and joins the rooms' move strings, walking the
// player from one room to the next. The other rooms' boxes are treated
// as walls while a room is being solved.
//
// Two prunings keep the BFS small, and neither can prune a solvable
// state:
//   - a "live" square: a box square from which a box can still be pushed
//     onto some goal (found by pulling backwards from the goals). A box
//     on a dead square, or pushed onto one, is lost.
//   - a frozen 2x2 block: four squares of walls and boxes holding a box
//     off its goal can never be broken apart.
//
// The search is bounded by a fixed number of states, never by the clock,
// so two runs always write the same bytes.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The most states one push search keeps before it gives up. */
const MAX_STATES = 250_000;

/** [dx, dy, walk letter, push letter], in a fixed order. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

type Level = {
  w: number;
  h: number;
  /** 1 is a wall (outside the board counts as a wall). */
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

type Room = {
  boxes: number[];
  goals: number[];
  done: boolean;
};

const at = (w: number, x: number, y: number): number => y * w + x;
const xs = (w: number, i: number): number => i % w;
const ys = (w: number, i: number): number => Math.floor(i / w);

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = at(w, x, y);
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Is (x, y) open floor (inside the board and not a wall)? */
function open(level: Level, x: number, y: number): boolean {
  return x >= 0 && y >= 0 && x < level.w && y < level.h && level.wall[at(level.w, x, y)] === 0;
}

/**
 * Live squares: every square a box can be pushed from onto some goal.
 * Start at the goals and pull backwards: a square y is live when the
 * square beyond it (y - d) is open, so a box on y could have been
 * pushed there from y - d in direction d.
 */
function liveSquares(level: Level): Uint8Array {
  const live = new Uint8Array(level.w * level.h);
  const queue: number[] = [];
  for (const g of level.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const cell = queue[qi]!;
    const cx = xs(level.w, cell);
    const cy = ys(level.w, cell);
    for (const [dx, dy] of DIRS) {
      const y = at(level.w, cx - dx, cy - dy);
      const beyond = at(level.w, cx - 2 * dx, cy - 2 * dy);
      if (!open(level, cx - dx, cy - dy) || !open(level, cx - 2 * dx, cy - 2 * dy)) continue;
      if (live[y]) continue;
      // The pull needs the player to stand on `beyond` while the box
      // sits on y; both squares must be open floor, checked above.
      void beyond;
      live[y] = 1;
      queue.push(y);
    }
  }
  return live;
}

/**
 * The squares the player reaches from `from` with `occupied` boxes in
 * the way. Returns the reached set as a stamp array plus the lowest
 * reached square, which names the whole region.
 */
function reach(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
): { low: number; stamp: number } {
  const stack: number[] = [from];
  seen[from] = stamp;
  let low = from;
  while (stack.length > 0) {
    const p = stack.pop()!;
    if (p < low) low = p;
    const px = xs(level.w, p);
    const py = ys(level.w, p);
    for (const [dx, dy] of DIRS) {
      const qx = px + dx;
      const qy = py + dy;
      if (!open(level, qx, qy)) continue;
      const q = at(level.w, qx, qy);
      if (occupied[q] || seen[q] === stamp) continue;
      seen[q] = stamp;
      stack.push(q);
    }
  }
  return { low, stamp };
}

/** A box just pushed to `t` sits in a frozen 2x2 block of walls/boxes. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const tx = xs(level.w, t);
  const ty = ys(level.w, t);
  for (const [ox, oy] of [[0, 0], [-1, 0], [0, -1], [-1, -1]] as const) {
    const cells = [
      at(level.w, tx + ox, ty + oy),
      at(level.w, tx + ox + 1, ty + oy),
      at(level.w, tx + ox, ty + oy + 1),
      at(level.w, tx + ox + 1, ty + oy + 1),
    ];
    let filled = true;
    let badBox = false;
    for (const c of cells) {
      const cx = xs(level.w, c);
      const cy = ys(level.w, c);
      const isWall = !open(level, cx, cy);
      if (!isWall && !occupied[c]) {
        filled = false;
        break;
      }
      if (occupied[c] && !goal[c]) badBox = true;
    }
    if (filled && badBox) return true;
  }
  return false;
}

/** Shortest walk from `from` to `to` around walls and boxes, lower-case. */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(level.w * level.h).fill(-1);
  const how: string[] = new Array(level.w * level.h);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    const px = xs(level.w, p);
    const py = ys(level.w, p);
    for (const [dx, dy, letter] of DIRS) {
      const qx = px + dx;
      const qy = py + dy;
      if (!open(level, qx, qy)) continue;
      const q = at(level.w, qx, qy);
      if (occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** Where the player stands after `moves` from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

/**
 * Push-optimal solution of one level (or one room as a level), by BFS
 * over pushes: {pushes, moves}, or null when the search gives up or
 * there is no solution. The first solved state a BFS meets is optimal.
 */
function solvePushOptimal(level: Level, maxStates: number = MAX_STATES): { pushes: number; moves: string } | null {
  const { w } = level;
  const size = w * level.h;
  const goal = new Uint8Array(size);
  for (const g of level.goals) goal[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  let stamp = 0;

  const statesBoxes: number[][] = [level.boxes.slice()];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const index = new Map<string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const startLow = reach(level, occupied, level.player, seen, ++stamp).low;
  for (const b of level.boxes) occupied[b] = 0;
  const keyOf = (boxes: number[], low: number): string => `${low}|${boxes.join(',')}`;
  index.set(keyOf(level.boxes, startLow), 0);

  let found = solved(level.boxes) ? 0 : -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const boxes = statesBoxes[head]!;
    const player = statesPlayer[head]!;
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    reach(level, occupied, player, seen, here);
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      const bx = xs(w, b);
      const by = ys(w, b);
      for (const [dx, dy] of DIRS) {
        const pushFrom = at(w, bx - dx, by - dy);
        const dest = at(w, bx + dx, by + dy);
        if (!open(level, bx - dx, by - dy) || seen[pushFrom] !== here) continue;
        if (!open(level, bx + dx, by + dy) || occupied[dest] || !live[dest]) continue;
        occupied[b] = 0;
        occupied[dest] = 1;
        const frozen = frozenBlock(level, occupied, goal, dest);
        let key = '';
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = dest;
          next.sort((a, b2) => a - b2);
          const low = reach(level, occupied, b, mark, ++stamp).low;
          key = keyOf(next, low);
        }
        occupied[dest] = 0;
        occupied[b] = 1;
        if (frozen || next === null || index.has(key)) continue;
        if (statesBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        index.set(key, statesBoxes.length);
        statesBoxes.push(next);
        statesPlayer.push(b);
        parent.push(head);
        if (solved(next)) {
          found = statesBoxes.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/**
 * The level's rooms: cut the dead squares of the player's row (the
 * hallway: a box there reaches no goal) out of the floor, and each
 * remaining part holding a box is a room. Null when some part holds
 * more or fewer boxes than goals, or a box starts in the hallway.
 */
function roomsOf(level: Level): Room[] | null {
  const live = liveSquares(level);
  const row = ys(level.w, level.player);
  const cut = new Uint8Array(level.w * level.h);
  for (let x = 0; x < level.w; x++) {
    const i = at(level.w, x, row);
    if (!level.wall[i] && !live[i]) cut[i] = 1;
  }
  if (level.boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(level.w * level.h).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < level.w * level.h; s++) {
    if (level.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      const px = xs(level.w, p);
      const py = ys(level.w, p);
      for (const [dx, dy] of DIRS) {
        const qx = px + dx;
        const qy = py + dy;
        if (!open(level, qx, qy)) continue;
        const q = at(level.w, qx, qy);
        if (cut[q] || part[q] >= 0) continue;
        part[q] = rooms.length;
        squares.push(q);
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of level.boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of level.goals) {
    if (part[g] < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * Solve a level room by room. Each room is solved with the other rooms'
 * boxes standing as walls (on their goals once their room is done);
 * the player walks on from where the last room left it. Falls back to
 * solving the whole level at once when it is no set of rooms.
 */
function solveByRooms(level: Level): { pushes: number; moves: string } | null {
  const rooms = roomsOf(level);
  if (rooms === null) return solvePushOptimal(level);
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushOptimal(sub);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

function solveLevel(text: string): string {
  try {
    const solved = solveByRooms(parseLevel(text));
    return solved === null ? '' : solved.moves;
  } catch {
    return '';
  }
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
