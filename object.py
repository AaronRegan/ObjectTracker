class Object:
    centerX = 0
    centerY = 0
    ID = ""

    # The class "constructor" - It's actually an initializer
    def __init__(self, centerX, centerY, ID):
        self.centerX = centerX
        self.centerY = centerY
        self.ID = ID

    def get_id(self):
        return self.ID

    def getcenter_x(self):
        return self.centerX

    def getcenter_y(self):
        return self.centerY

    def __str__(self):
        return "Center location:(%d,%d) ID:%s" % (self.centerX, self.centerY, self.ID)
