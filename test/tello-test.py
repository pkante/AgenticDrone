from djitellopy import Tello
import cv2

class TelloLLM():
    def __init__(self):
        self.drone = Tello()
        self.drone.connect()
        self.battery = self.drone.query_battery()
        
    def check_battery(self):
        self.battery = self.drone.query_battery()
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 30:
            print('is too low [WARNING]')
        else:
            print('[OK]')
            return True
        return False

    def start(self):
        if not self.check_battery():
            return
        
        return
        
        self.streamOn = True
        self.drone.takeoff()
        self.drone.move_up(20)
        self.drone.streamon()
        print("> Application Start")

        frame_read = self.drone.get_frame_read()

        count = 0
        while (True):
            frame = frame_read.frame
            if frame is None:
                continue
            print("### GET Frame: ", frame.shape)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"./cache/frame_{count}.png", frame)
            cv2.imshow("Tello", frame)
            key = cv2.waitKey(10) & 0xff
            # Press esc to exit
            if key == 27:
                break
            count += 1
        self.drone.streamoff()
        self.drone.land()

def main():
    tello = TelloLLM()
    tello.start()

if __name__ == "__main__":
    main()
