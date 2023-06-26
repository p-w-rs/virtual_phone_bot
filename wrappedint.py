# You are Nina Powers. Your job is to respond to telemarketers as if in verbal communication. You are enthusiastic about the conversation. Your ultimate goal is to figure out the company the telemarketer works for and from which state thier company is based. After you figure this out ask them to put you on a do not call list and say goodbye.
# Hello! This is Nina!


class WrappedInt:
    def __init__(self, value, m):
        self.m = m
        self.value = self.wrap(value)
        self.first = self.value

    def wrap(self, n):
        return n % self.m if n >= 0 else (n % self.m + self.m) % self.m

    def pp(self):
        ret = self.value
        self.value = self.wrap(self.value + 1)
        return ret

    def reset(self):
        self.value = self.first

    def __index__(self):
        return self.value

    def __repr__(self):
        return str(self.value)

    def __add__(self, other):
        return WrappedInt(self.value + other, self.m)

    def __radd__(self, other):
        return WrappedInt(other + self.value, self.m)

    def __sub__(self, other):
        return WrappedInt(self.value - other, self.m)

    def __rsub__(self, other):
        return WrappedInt(other - self.value, self.m)

    def __mul__(self, other):
        return WrappedInt(self.value * other, self.m)

    def __rmul__(self, other):
        return WrappedInt(other * self.value, self.m)

    def __truediv__(self, other):
        return WrappedInt(self.value // other, self.m)

    def __rtruediv__(self, other):
        return WrappedInt(other // self.value, self.m)

    def __floordiv__(self, other):
        return WrappedInt(self.value // other, self.m)

    def __rfloordiv__(self, other):
        return WrappedInt(other // self.value, self.m)

    def __mod__(self, other):
        return WrappedInt(self.value % other, self.m)

    def __rmod__(self, other):
        return WrappedInt(other % self.value, self.m)

    def __divmod__(self, other):
        q, r = divmod(self.value, other)
        return WrappedInt(q, self.m), WrappedInt(r, self.m)

    def __pow__(self, other):
        return WrappedInt(pow(self.value, other), self.m)

    def __eq__(self, other):
        if isinstance(other, WrappedInt):
            return self.m == other.m and self.value == other.value
        return self.value == other

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other
