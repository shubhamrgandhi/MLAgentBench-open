# This script is used to calculate the area of a triangle.

def calculate_area_triangle(base, height):
  """Calculates the area of a triangle.

  Args:
    base: The base of the triangle in inches.
    height: The height of the triangle in inches.

  Returns:
    The area of the triangle in square inches.
  """

  # Calculate the area of the triangle.
  area = 0.5 * base * height

  # Return the area of the triangle.
  return area


# Get the base and height of the triangle from the user.
base = float(input("Enter the base of the triangle in inches: "))
height = float(input("Enter the height of the triangle in inches: "))

# Calculate the area of the triangle.
area = calculate_area_triangle(base, height)

# Print the area of the triangle.
print("The area of the triangle is", area, "square inches.")