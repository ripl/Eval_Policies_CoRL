"""Constants for the SimplerEnv WidowX / Bridge overfitting experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


OFFICIAL_OBJ_EPISODE_RANGE = (0, 24)


@dataclass(frozen=True)
class WidowXBridgeTask:
    """One SimplerEnv WidowX / Bridge task used by the CogACT script."""

    env_name: str
    short_name: str


TASKS = (
    WidowXBridgeTask(
        env_name="StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        short_name="stack_green_cube_on_yellow_cube",
    ),
    WidowXBridgeTask(
        env_name="PutCarrotOnPlateInScene-v0",
        short_name="put_carrot_on_plate",
    ),
    WidowXBridgeTask(
        env_name="PutSpoonOnTableClothInScene-v0",
        short_name="put_spoon_on_tablecloth",
    ),
    WidowXBridgeTask(
        env_name="PutEggplantInBasketScene-v0",
        short_name="put_eggplant_in_basket",
    ),
)


def task_names() -> list[str]:
    return [task.env_name for task in TASKS]


def as_config() -> dict[str, object]:
    return {
        "suite": "simplerenv_widowx_bridge",
        "official_obj_episode_range": list(OFFICIAL_OBJ_EPISODE_RANGE),
        "tasks": [asdict(task) for task in TASKS],
    }


if __name__ == "__main__":
    print(json.dumps(as_config(), indent=2))

