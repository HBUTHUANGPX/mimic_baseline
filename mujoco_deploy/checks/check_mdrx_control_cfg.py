import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def main():
    from deploy.utils.cfg import cfg, select_robot

    select_robot("mdrx")

    assert cfg.leg_P_gains == [80.0, 80.0, 80.0, 80.0, 30.0, 30.0] * 2
    assert cfg.leg_D_gains == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0] * 2
    assert cfg.leg_tq_max == [62.4, 62.4, 19.5, 62.4, 39.0, 39.0] * 2

    assert cfg.pelvis_P_gains == [80.0, 80.0, 80.0]
    assert cfg.pelvis_D_gains == [2.0, 2.0, 2.0]
    assert cfg.pelvis_tq_max == [62.4, 39.0, 39.0]

    assert cfg.arm_P_gains == [15.0] * (6 * 2)
    assert cfg.arm_D_gains == [1.5, 1.5, 1.0, 1.5, 1.0, 1.0] * 2
    assert cfg.arm_tq_max == [19.5] * (6 * 2)

    assert len(cfg.leg_tq_max + cfg.pelvis_tq_max + cfg.arm_tq_max) == 27


if __name__ == "__main__":
    main()
