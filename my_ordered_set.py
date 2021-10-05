class ordered_set:
    def __init__(self, size):
        self.set = []
        self.size = size
        self.curr_size = 0
    
    def push(self, pair):
        if self.curr_size == 0:
            self.set.append(pair)
            self.curr_size += 1
        elif self.curr_size < self.size:
            self.set.append(pair)
            self.curr_size += 1
            self.quicksort(0, self.curr_size - 1)
        else:
            if pair[1] > self.set[self.size - 1][1]:
                self.set[self.size - 1] = pair
                self.quicksort(0, self.size - 1)

    def quicksort(self, first, last):
        temp = 0
        if (last - first) < 1:
            return
        pivot = self.median_of_three(first, last)
        pivot = self.partition(first, last, pivot)
        self.quicksort(first, pivot - 1)
        self.quicksort(pivot + 1, last)
    
    def median_of_three(self, left, right):
        middle = (left + right) / 2
        return int(middle)
    
    def partition(self, left, right, pivot_index):
        #swap the first value with the median value
        temp = self.set[left]
        self.set[left] = self.set[pivot_index]
        self.set[pivot_index] = temp
        
        #set bounds for up and down
        up = left + 1
        down = right

        done = False

        while done == False:
            while (self.set[up][1]) >= self.set[left][1] and (up < right):
                up += 1
            while (self.set[down][1] < self.set[left][1]) and (down > left):
                down -= 1
            if up < down:
                temp = self.set[up]
                self.set[up] = self.set[down]
                self.set[down] = temp
            else:
                done = True
        
        #put pivot value back where it belongs and replace it with down
        temp = self.set[left]
        self.set[left] = self.set[down]
        self.set[down] = temp

        return down

        



    