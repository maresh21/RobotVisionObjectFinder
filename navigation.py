def decide_movement(
    object_x,
    image_width,
    object_area,
    stop_area=15000
):
    """
    Decide how the virtual robot should move
    based on the target object's position.
    """

    # Calculate the center of the target
    image_center = image_width / 2

    # Calculate distance from image center
    difference = object_x - image_center

    # Check if target is close enough
    if object_area >= stop_area:
        return "STOP"

    # Target is on the left
    elif difference < -100:
        return "TURN LEFT"

    # Target is on the right
    elif difference > 100:
        return "TURN RIGHT"

    # Target is approximately in the center
    else:
        return "MOVE FORWARD"