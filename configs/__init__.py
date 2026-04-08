from .lift_manip_shirt import LiftManipShortShirt


def load_task_from_name(task_name):
    if task_name == "lift_manip_shirt":
        cfg = LiftManipShortShirt()
        task = cfg.finalize()
    else:
        print(f"Unknow Taks:{task_name}")
        return 0

    return task