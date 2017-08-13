import numpy as np
class Line():
    def __init__(self, n):
        self.max_frames = n # max number of frame's line to store
        self.previous_lines = []
        self.store_counter = 0
        self.l_fit_avg = []
        self.r_fit_avg = []
        self.cur_avg = []
        self.area_avg = []
        self.l_fit_last = []
        self.r_fit_last = []
        self.last_cur = []
        self.last_area = []
    
    def avg(previous_lines, counter):
        l_fit1 = []
        l_fit2 = []
        l_fit3 = []
        
        r_fit1 = []
        r_fit2 = []
        r_fit3 = [] 
        cur = [] 
        area = []      
        
        for i in range(counter):
            l_fit1.append(previous_lines[i][0][0])
            l_fit2.append(previous_lines[i][0][1])
            l_fit3.append(previous_lines[i][0][2])
            
            r_fit1.append(previous_lines[i][1][0])
            r_fit2.append(previous_lines[i][1][1])
            r_fit3.append(previous_lines[i][1][2])
            
            cur.append(previous_lines[i][2])
            area.append(previous_lines[i][3])
            
        l_fit_avg = np.array([np.mean(l_fit1), np.mean(l_fit2), np.mean(l_fit3)])
        r_fit_avg = np.array([np.mean(r_fit1), np.mean(r_fit2), np.mean(r_fit3)])
        cur_avg = np.mean(cur)
        area_avg = np.mean(area)
        return l_fit_avg, r_fit_avg, cur_avg, area_avg
    
    def store(self, l_fit, r_fit, cur, area):
        if self.store_counter < self.max_frames:
            self.store_counter += 1
            self.previous_lines.append([l_fit, r_fit, cur, area])
        else:
            self.store_counter = self.max_frames
            self.previous_lines = self.previous_lines[1:]
            self.previous_lines.append([l_fit, r_fit, cur, area])
        
        self.l_fit_avg, self.r_fit_avg, self.cur_avg, self.area_avg = Line.avg(self.previous_lines, self.store_counter)
        
    def last(self, l_fit, r_fit, cur, area):        
        self.l_fit_last = l_fit
        self.r_fit_last = r_fit
        self.last_cur = cur
        self.last_area = area
        
    def check(self, cur, area, thres= 0.2):
        check_pass = True
        area_delta = abs(self.last_area - area)/self.last_area
        cur_delta = abs(self.last_cur - cur)/self.last_cur
        if area_delta > thres:
            check_pass = False
        if cur_delta > 2:
            check_pass = False
        if abs(cur) < 300:
            check_pass = False
        return check_pass
