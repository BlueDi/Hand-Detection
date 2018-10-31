# Hand Detection with OpenCV

Hand detection is a simple idea, but it requires some advanced tools to do it.

In this project we will allow the user to set the area that will be used to determine the color to capture, and then show, over the videocam feed, where the hand is and what signals it is doing.

The available signals are:

- Numbers using the fingers from 0 to 5;
- OK signal;
- Cool/Fixe signal.

## Usage

To start the project run at root:
`python hand_detection.py <optional_param>`

- `-h` can be used to know what the available params are;
- `--left` or `-l`: Moves the ROI to the left side;
- `--shot` or `-s`: Detects hand in a single shot, a picture, not video.

Then the webcam image is shown saying 'Welcome'.

You can then:

- Press `h` to go to the color calculator;
- Or press `v` to go to the hand detection;

Press `q` at any time to close the project.

#### Hand Contours

In this mode, you will calculate the color of the hand to be detected.

Place your hand in the _blue square_ and then

- Press `v` to capture;

If the result pleases you

- Press `any key` to proceed to video hand detection.
  If it doesn't:
- Press `n` to return to the color calculator.

#### Hand Detection

In this mode, you will be able to do signs with a single hand that will be analysed by the algorithm.

Check the list of available signs above.

To exit the app press `q`.
