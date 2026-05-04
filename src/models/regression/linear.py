def cost_function(x,y,z,b):
        m = len(x)
        cost_sum = 0

        for i in range(m):
            f = z * x[i] + b
            cost = (f - y[i]) ** 2
            cost_sum += cost

        total_cost = (1 / (2 * m)) * cost_sum
        return total_cost

def gradient_function(x,y,z,b):
        m = len(x)
        dz_sum = 0
        db_sum = 0

        for i in range(m):
            f = z * x[i] + b
            dz_sum += (f - y[i]) * x[i]
            db_sum += (f - y[i])

        dz = (1 / m) * dz_sum
        db = (1 / m) * db_sum

        return dz, db
    
def gradient_descent(x, y, alpha, iterations):
        w = 0
        b = 0

        for i in range(iterations):
            dz, db = gradient_function(x, y, w, b)

            w -= alpha * dz
            b -= alpha * db

            # print(f"Iteration {i+1}: Cost = {cost_function(x, y, w, b)}, w = {w}, b = {b}")

        return w, b