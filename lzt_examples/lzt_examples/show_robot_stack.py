"""Print the data flow through the mechanical-arm algorithm stack."""


def main() -> None:
    stages = (
        ("perception", "RGB-D -> Top-K object poses"),
        ("grasp_planning", "object poses -> robust grasp pose"),
        ("kinematics", "tool pose -> joint solution"),
        ("motion_planning", "joint states -> collision-free path"),
        ("trajectory", "path -> timed position/velocity/acceleration"),
        ("control", "trajectory -> actuator commands"),
        ("backend", "MuJoCo now; real robot adapter later"),
    )
    for name, contract in stages:
        print(f"{name}: {contract}")


if __name__ == "__main__":
    main()

