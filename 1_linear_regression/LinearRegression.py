from numpy import *

def compute_error_for_line_given_points(b,m,points):
	# Initialize error as 0
	totalError = 0

	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		totalError += (y- (m*x + b))**2
	return totalError/float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m

	# Gradient Descent
	for i in range(num_iterations):
		# Update b and m using gradient step
		b, m = step_gradient(b,m, array(points), learning_rate)

	return [b,m]

def step_gradient(b_current, m_current, points, learning_rate):
	
	b_gradient = 0
	m_gradient = 0

	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		N = float(len(points))

		b_gradient += (-2/N) * (y - (m_current * x + b_current))
		m_gradient += (-2/N) * (x * (y - (m_current * x + b_current)))

	# Update b and m
	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)
	return [new_b,new_m]

def run():
	# Step 1 - collect data
	points = genfromtxt('data.csv',delimiter=',')

	# Step 2 - Define Hyper parameters
	# How fast should our model converge
	learning_rate = 0.0001
	# y = mx + b
	initial_b = 0
	initial_m = 0
	num_iterations = 1000

	# Step 3 - Train model
	print 'Starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b,initial_m,compute_error_for_line_given_points(initial_b,initial_m,points))
	[b, m] = gradient_descent_runner(points,initial_b,initial_m,learning_rate,num_iterations)

	print 'Ending point at b = {1}, m = {2}, error = {3}'.format(num_iterations,b,m,compute_error_for_line_given_points(b,m,points))

	x = 22.9
	prediction = m * x + b
	print 'Prediction on {0} results in {1}'.format(x,prediction)

if __name__ == '__main__':
	run()