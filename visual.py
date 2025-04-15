import matplotlib.pyplot as plt

# Read data from the file
with open("total_errors.txt", "r") as file:
    data = file.read()

# Split the data by newline characters
lines = data.strip().split("\n")

# Extract y values
y_values = [float(line.rstrip(', ')) for line in lines]

# Extract x values
x_values = range(1, len(y_values) + 1)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker="o", linestyle="-", color="b")
plt.xlabel("Line Number")
plt.ylabel("Value")
plt.title("Total Errors")
plt.grid(True)

# Show the plot
plt.show()