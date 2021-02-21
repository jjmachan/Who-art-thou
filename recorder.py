import cv2
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-o', '--output',
            help='The name of the file to save recoded video to',
            default='output.avi'
            )

    args = parser.parse_args()
    cap = cv2.VideoCapture(1)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, 20.0, (640,480))

    while True:
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 1)
            out.write(frame)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) &0xFF == ord('q'):
                break

        else:
            break

    print('Saved to: ', args.output)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
