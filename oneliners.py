
def list_duplicates(arr):
    return list(set(arr))


def maxprod(nums):
    return (sorted(nums)[-1] - 1) * (sorted(nums)[-2] - 1)


def zerosum(n):
    return sorted(([] if n % 2 == 0 else [0]) + [item for sublist in [[i + 1, -(i + 1)] for i in range(n // 2)] for item in sublist])


def zerosum2(n):
    return [i for i in range(n-1)] + [-(n-1)*(n-2)/2]


def evendigits(arr):
    # return len(list(filter(lambda x: len(str(x)) % 2 == 0, arr)))
    return len([x for x in arr if len(str(x)) % 2 == 0])


def charinstring(string, target_ch):
    return len([ch for ch in string if ch == target_ch])


arr = [1, 2, 1, 1, 3]
# print(list_duplicates(arr))
# print(maxprod([3, 4, 5, 2]))
# print(zerosum2(6))
# print(evendigits([12, 345, 2, 6, 7896]))
print(charinstring('hey there baby', 'e'))
