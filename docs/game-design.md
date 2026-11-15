# ALL-ANIME OPEN WORLD FIGHTER — Design Doc

**Working title:** The Open World of Anime
**Creator:** Ethan Bennett (`666max66_maxy` / Zenth) · Group: 0xScripts
**Engine:** Roblox Studio (Luau)
**Status:** Pre-production — concept + first move specced

---

## 1. The Pitch

An **open-world Roblox fighting game** where characters and moves from **across anime** share one connected world. Players roam a persistent map, pick their fighter, and fight using signature techniques pulled from the shows they grew up on.

Not a lobby fighter. Not a "pick a character, load an arena" game. You **walk the world**, meet other players, and fight them anywhere.

**Why players would care:** every anime fan has imagined their favorite character's move in a game. Most Roblox anime games are either clicker/simulator grinders or rigid arena fighters. An open world with real combat controls is the gap.

**First anime: Jujutsu Kaisen (decided).** The launch zone, the launch roster, and the flagship ultimate (Hollow Nuke) are all JJK. Cursed energy is the combat system's foundation — it powers techniques, and running out of it is a real loss condition.

---

## 2. Design Pillars

1. **Signature moves feel like the anime.** If a move is famous, it must land with the right charge-up, colors, and payoff. Feel > balance.
2. **The world is the content.** Exploration, hidden spots, roaming players. Fighting happens wherever people meet.
3. **Easy to start, deep to master.** Anyone can throw a punch in 10 seconds; combos and cancellable moves take real practice.
4. **Crossover, not clones.** Characters from different anime must feel mechanically different — not reskins.

---

## 3. The World

- **Shared open map**, one server holds a slice of it (Roblox servers cap players, so the world loads in zones).
- **Zones themed per anime** — zone one is JJK: a modern Japanese city by night, cursed spirits in the alleys, a school campus as the safe hub. Later zones expand outward to other series.
- **Free-roam PvP**, with safe zones for trading/socializing.
- **Progression lives in the world**: find moves in hidden places, train with NPCs, unlock traversal (dashes, wall-runs, flight) to reach new zones.

**Traversal is core, not a bonus.** Open worlds live or die on how good it feels to move through them.

---

## 4. Combat System

**Camera:** third-person over-the-shoulder. Lock-on toggle for duels.

**Core loop per fight:**
`approach → poke/whiff-punish → combo → knockback → re-engage`

| System | Notes |
|---|---|
| Light attack | Fast, low damage, chains into combos (M1 chain, 3–5 hits) |
| Heavy attack | Slower, guard-breaks, launches |
| Block / parry | Hold to block, timed parry to stun attacker |
| Dash / dodge | Short i-frame dodge on cooldown |
| Moves (slots 1–4) | Character-specific signature techniques, each with own cooldown |
| Ultimate | One per character, big charge, huge payoff (see Hollow Nuke) |
| Stamina | Limits dashing + blocking so fights don't turn into dodge-spam |

**Combat rules that make it feel good:**
- Every attack has **startup → active → recovery** so it can be reacted to and punished.
- Hits **stun briefly** (hitstun) so combos can connect.
- **Knockback** resets neutral so fights don't become infinite combo loops.
- Ultimate moves are **telegraphed** — long charge, loud VFX. You can dodge or interrupt them.

---

## 5. Signature Move Spec — "HOLLOW NUKE"

> *The character floats up while spinning a little, then does a blue, then a red, pulls them together, and a big purple explosion happens.*

This is the flagship move. It gets built first, because if this feels incredible, the game has a soul.

### Phase breakdown

| # | Phase | Duration | What happens | Counterplay |
|---|---|---|---|---|
| 1 | **Float & Spin** | 1.2s | Character rises ~6 studs off the ground, slowly spinning, arms out. Faint purple aura builds. HUD ring fills. | Fully interruptible — any hit or stun cancels it |
| 2 | **Blue** | 0.6s | Blue orb spawns in the right hand. It hums and pulls nearby particles inward. | Still interruptible |
| 3 | **Red** | 0.6s | Red orb spawns in the left hand, pulsing, repelling particles. | Still interruptible |
| 4 | **Pull Together** | 0.5s | Both hands slam together at chest height. Screen shake begins, particles orbit the convergence point, color shifts toward violet. | **Last window to dodge** |
| 5 | **Purple Detonation** | 1.0s impact + 0.8s recovery | Massive purple sphere expands from the character, then collapses inward and fades. Heavy screen shake, bloom flash, ground scorch decal. | Cannot be cancelled — but the user is stuck in 0.8s recovery |

**Total cast time:** ~2.9s · **Cooldown:** long (ultimate-tier) · **Damage:** very high, falls off with distance from center

### Visual checklist
- Rising = `TweenService` on HumanoidRootPart Y position, with spin = `CFrame.Angles` on Y axis
- Blue orb = `Sphere` mesh + `PointLight` (blue) + `ParticleEmitter` sucking particles inward
- Red orb = same, red light, particles pushing outward
- Convergence = particles orbiting, light blending blue+red into purple
- Detonation = expanding `Sphere` with transparency tween, purple `PointLight` scaled up fast, `Bloom` post-process flash, screen shake via camera offset, scorch `Decal` on floor

### Code shape (Luau, pseudo)
```lua
local function hollowNuke(character)
    local root = character.HumanoidRootPart

    -- 1. float & spin (interruptible)
    if not charge(root, 1.2, "FloatSpin") then return end
    -- 2. blue / 3. red (interruptible)
    spawnOrb(root, "Blue", 0.6)
    if not charge(root, 0.6, "Blue") then return end
    spawnOrb(root, "Red", 0.6)
    if not charge(root, 0.6, "Red") then return end
    -- 4. pull together (last dodge window)
    convergeOrbs(root, 0.5)
    -- 5. detonate (unstoppable)
    detonate(root, radius = 40, damage = 85)
end
```

`charge()` yields and returns `false` if the caster gets stunned — that's what makes phases 1–4 cancellable.

### Why this is a good move design
- **Telegraphed.** 2.9 seconds is long. Anyone watching can run or interrupt — so landing it feels earned.
- **Readable.** Blue, then red, then purple: the sequence itself communicates "get out of the way."
- **High risk, high reward.** Enormous damage, but you're a floating target for three seconds and helpless for 0.8s after.
- The **spin+float** phase is what makes it cinematic. Nobody else on Roblox has that silhouette.

---

## 6. Scope Reality Check

**Honest truth:** multi-anime + open world + combat = a big game. Bigger than one person should attempt as their first serious project. Studios with teams ship these.

**So we build a vertical slice first** — small but *complete*, to prove the feel:

### Milestone 1 — "The Duel" (target: first playable)
- One small JJK-themed arena (a destroyed city block)
- One JJK character with basic moves: light chain, heavy, block, dash
- **Hollow Nuke fully built and polished**
- Cursed energy resource tied to techniques
- Two players can fight each other and it feels good

If The Duel is fun, everything else is expansion. If it isn't fun, adding an open world won't save it.

### Milestone 2 — "The First Zone"
- JJK city zone, walkable, with traversal
- 2 characters (two different JJK fighters, mechanically distinct)
- Free-roam PvP between players in that zone

### Milestone 3 — "The World"
- 3+ zones, connected
- 4+ characters, each with 4 moves + ultimate
- Progression, unlocks, hidden moves

### Milestone 4 — Live game
- Monetization, seasons, new characters, community events

---

## 7. Build Order (what to do first)

1. **Combat prototype** — no art. Two gray blocks, light chain + hitstun + knockback. Does punching feel good?
2. **Hollow Nuke** — built on the prototype character. All 5 phases, full VFX.
3. **Duel arena** — one map, one spawn, round system.
4. **Playtest with other people.** Fix what feels bad. Repeat until it's fun.
5. *Then* open-world and extra characters.

---

## 8. Open Questions

- ~~Which anime first?~~ **JJK. Decided.**
- Lock-on combat or free-camera fighting?
- How many moves per character: 4 or 6?
- Movement: grounded + dash, or flight allowed in open world?
- ~~Working title?~~ **The Open World of Anime (working title, decided).**
- Which JJK characters make the launch roster, and what are their four moves each?
- Domains: playable ultimate abilities, or cinematic finishers only?

---

*This doc is a living document. Every decision here can change once the prototype teaches us something.*
