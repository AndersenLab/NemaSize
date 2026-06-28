import sys

import cv2
import numpy

window = 200
true_diameter = 2010

def main():
    img = cv2.imread(sys.argv[1], cv2.IMREAD_ANYDEPTH)
    img = ((img >> 4) & (2**8-1)).astype(numpy.uint8)
    data = img.astype(numpy.float32)
    mids = numpy.zeros(2, numpy.int32)

    try:
        N = img.shape[0]
        N2 = N // 2
        x0 = best_offset(data[:, N2 + 350:N2 + 450])
        x1 = best_offset(data[:, N2 - 450:N2 - 350])
        mids[0] = int(numpy.round((x0 + x1) / 2))
        x0 = best_offset(data[N2 + 350:N2 + 450, :].T)
        x1 = best_offset(data[N2 - 450:N2 - 350, :].T)
        mids[1] = int(numpy.round((x0 + x1) / 2))

        x00 = best_offset(data[:window, mids[1] + 400])
        x10 = N - 1 - best_offset(data[-window:, mids[1] - 400][::-1])
        x01 = best_offset(data[:window, mids[1] - 400])
        x11 = N - 1 - best_offset(data[-window:, mids[1] + 400][::-1])
        r0 = ((x10 - x00) ** 2 + 800 ** 2) ** 0.5
        r1 = ((x11 - x01) ** 2 + 800 ** 2) ** 0.5

        x00 = best_offset(data[mids[0] + 400, :window])
        x10 = N - 1 - best_offset(data[mids[0] - 400, -window:][::-1])
        x01 = best_offset(data[mids[0] - 400, :window])
        x11 = N - 1 - best_offset(data[mids[0] + 400, -window:][::-1])
        r2 = ((x10 - x00) ** 2 + 800 ** 2) ** 0.5
        r3 = ((x11 - x01) ** 2 + 800 ** 2) ** 0.5

        radius = numpy.median(numpy.r_[r0, r1, r2, r3]) / 2 - 5
        mask = numpy.ones(img.shape, bool)
        mask[numpy.where(((numpy.arange(img.shape[0]) - mids[0]).reshape(-1, 1) ** 2 + (numpy.arange(img.shape[0]) - mids[1]).reshape(1, -1) ** 2) ** 0.5 > radius)] = False
        img1 = numpy.zeros((img.shape[0], img.shape[1], 3), numpy.uint8)
        img1[:, :, :] = img[:, :, numpy.newaxis] * mask[:, :, numpy.newaxis]
        cv2.imwrite(sys.argv[2], img1)
    except:
        print(sys.argv[1:3])


def best_offset(data):
    slope = data[:-1] - data[1:]
    best = numpy.argmax(slope)
    return best


main()