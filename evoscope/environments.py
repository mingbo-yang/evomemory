"""Public observations and reset/replay environments; no privileged editor state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .core import Task


@dataclass(frozen=True)
class View:
    observation: str
    admissible: tuple[str, ...]
    score: float = 0.0
    done: bool = False


class Environment(Protocol):
    def reset(self, task: Task) -> View: ...
    def step(self, action: str) -> View: ...
    def close(self) -> None: ...


class ToyEnvironment:
    """Deterministic plumbing fixture, NOT a research benchmark.

    Tasks differ in visible cleanliness. A broad 'place now' policy helps with
    clean objects but harms dirty objects. This makes scope repair testable.
    """

    def reset(self, task: Task) -> View:
        self.clean = bool(task.payload.get("clean", False))
        self.done = False
        self.score = 0.0
        self.label = task.payload.get("object", "cup")
        return self._view()

    def _view(self) -> View:
        return View(f"The {self.label} is {'clean' if self.clean else 'dirty'}. Destination: shelf.",
                    () if self.done else ("clean", "place", "wait"), self.score, self.done)

    def step(self, action: str) -> View:
        if self.done:
            raise RuntimeError("step after termination")
        if action == "clean":
            self.clean = True
        elif action == "place":
            self.done = True
            self.score = float(self.clean)
        elif action != "wait":
            raise ValueError("invalid action")
        return self._view()

    def close(self) -> None:
        pass


class AlfWorldEnvironment:
    """One deterministic TextWorld game per task, restored by reset+action replay.

    Uses ALFWorld's name demangler with shuffling disabled, no expert wrapper.
    Imports optional dependencies only when actually running ALFWorld.
    """

    action_protocol = "alfworld-native-v1"

    _registrations: dict[tuple[str, int], str] = {}

    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self.env = None

    def reset(self, task: Task) -> View:
        from pathlib import Path
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        self.close()
        game = str(Path(task.payload["gamefile"]).expanduser().resolve(strict=True))
        key = (game, self.max_steps)
        if key not in self._registrations:
            self._registrations[key] = textworld.gym.register_games(
                [game], textworld.EnvInfos(won=True, admissible_commands=True,
                                          extras=["gamefile"]),
                batch_size=1, asynchronous=False, max_episode_steps=self.max_steps,
                wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        self.env = textworld.gym.make(self._registrations[key])
        self.env.seed(task.seed)
        observations, infos = self.env.reset()
        return self._view(observations, infos, False)

    @staticmethod
    def _view(observations, infos, done) -> View:
        return View(str(observations[0]), tuple(sorted(infos["admissible_commands"][0])),
                    float(bool(infos.get("won", [False])[0])), bool(done))

    def step(self, action: str) -> View:
        observations, _, dones, infos = self.env.step([action])
        return self._view(observations, infos, dones[0])

    def close(self) -> None:
        if self.env is None:
            return
        env, self.env = self.env, None
        # TextWorld 1.7 PddlEnv inherits a no-op close. Its dlopen'ed,
        # unlinked 43 MiB planner otherwise stays allocated until process exit.
        # Inspect actual attributes: Wrapper.__getattr__ forwards attributes.
        pending, visited, planners = [env], set(), []
        while pending:
            node = pending.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            attrs = vars(node) if hasattr(node, "__dict__") else {}
            if "downward_lib" in attrs:
                planners.append((node, attrs["downward_lib"]))
            for name in ("batch_env", "_wrapped_env", "env"):
                child = attrs.get(name)
                if child is not None:
                    pending.append(child)
            pending.extend(attrs.get("envs", []) or [])
        try:
            env.close()
        finally:
            if planners:
                import fast_downward
                released = set()
                for node, lib in planners:
                    if lib is not None and id(lib) not in released:
                        fast_downward.close_lib(lib)
                        released.add(id(lib))
                    node.downward_lib = None


class WebShopEnvironment:
    """Official local WebShop simulator, human goals, reset/replay by goal index.

    Catalogs and Lucene indexes are shared read-only. Sessions are cleared on
    reset: completed purchases must not contaminate paired replay branches.
    """
    action_protocol = "webshop-native-v1"

    _servers: dict = {}

    def __init__(self, root: str, num_products: int = 1000, catalog_seed: int = 42):
        from pathlib import Path
        self.root = str(Path(root).resolve(strict=True))
        self.num_products, self.catalog_seed = num_products, catalog_seed
        self.env = None

    def server(self):
        import random
        import sys
        import numpy as np
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        from web_agent_site.envs.web_agent_text_env import SimServer
        key = (self.root, self.num_products, self.catalog_seed)
        if key not in self._servers:
            state, np_state = random.getstate(), np.random.get_state()
            try:
                random.seed(self.catalog_seed)
                np.random.seed(self.catalog_seed)
                self._servers[key] = SimServer('http://127.0.0.1:3000',
                    self.root + '/data/items_shuffle_1000.json',
                    num_products=self.num_products, human_goals=True)
            finally:
                random.setstate(state)
                np.random.set_state(np_state)
        return self._servers[key]

    def reset(self, task: Task) -> View:
        server = self.server()
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        index = task.payload['goal_index']
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(server.goals):
            raise ValueError('invalid WebShop goal index')
        server.user_sessions.clear()
        self.env = WebAgentTextEnv(observation_mode='text', server=server)
        server.user_sessions.clear()
        observation, _ = self.env.reset(session=index)
        import random
        random.seed(task.seed)
        return self._view(observation, 0.0, False)

    def _view(self, observation, score, done):
        if done:
            # Upstream HTML text extraction includes CSS-hidden target answers.
            # Keep the public completion message; numeric reward stays in View.
            return View('Thank you for shopping with us! Purchase completed.', (), float(score), True)
        available = self.env.get_available_actions()
        actions = [f'click[{v}]' for v in available['clickables'] if v != 'search']
        # Official text simulator accepts search on every nonterminal page.
        actions.append('search[<keywords>]')
        return View(str(observation), tuple(sorted(actions)), float(score), False)

    def step(self, action: str) -> View:
        observation, score, done, _ = self.env.step(action)
        return self._view(observation, score, done)

    def close(self) -> None:
        if self.env is not None:
            self.env.server.user_sessions.pop(self.env.session, None)
            self.env.close()
            self.env = None
