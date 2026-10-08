from navigation import decide_movement


image_width = 1000


# Test 1: Target on the left
action = decide_movement(
    object_x=200,
    image_width=image_width,
    object_area=5000
)

print("Target LEFT →", action)


# Test 2: Target in the center
action = decide_movement(
    object_x=500,
    image_width=image_width,
    object_area=5000
)

print("Target CENTER →", action)


# Test 3: Target on the right
action = decide_movement(
    object_x=800,
    image_width=image_width,
    object_area=5000
)

print("Target RIGHT →", action)


# Test 4: Target is very close
action = decide_movement(
    object_x=500,
    image_width=image_width,
    object_area=20000
)

print("Target CLOSE →", action)