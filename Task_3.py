import numpy as np

# Step 1: Define custom dtype for a person
person_dtype = np.dtype([
    ('name', 'U20'),   # Unicode string (max 20 characters)
    ('age', 'i4'),     # Integer (32-bit)
    ('height', 'f4')   # Float (32-bit)
])

# Step 2: Ask user for number of entries
n = int(input("How many people do you want to enter? "))

# Step 3: Create an empty structured array
people = np.empty(n, dtype=person_dtype)

# Step 4: Get user input for each person
print("\nEnter person details (name, age, height in cm):")
for i in range(n):
    name = input(f"Person {i + 1} - Name: ")
    age = int(input(f"Person {i + 1} - Age: "))
    height = float(input(f"Person {i + 1} - Height (cm): "))
    people[i] = (name, age, height)

# Step 5: Print the structured array
print("\nStructured NumPy Array:\n", people)

# Step 6: Access and print specific fields
print("\nNames:", people['name'])
print("Ages:", people['age'])
print("Heights:", people['height'])
