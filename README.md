# Hand Detection with OpenCV

Hand detection is a simple idea, but it requires some advanced tools to do it.

In this project we will allow the user to set the area that will be used to determine the color to capture, and then show, over the videocam feed, where the hand is and what signals it is doing.

The available signals are:

-   Numbers using the fingers from 0 to 5;
-   OK signal;
-   Cool/Fixe signal.

<img src="https://user-images.githubusercontent.com/9451036/52899416-d2781b00-31e1-11e9-95fd-eaddab198953.png" alt="Detection of numbers from 0 to 5" width="500" />

<img src="https://user-images.githubusercontent.com/9451036/52899444-3ac6fc80-31e2-11e9-8e4a-62c029e90061.png" alt="Detection of 'ok' and 'cool' gestures" width="500" />

## Usage

To start the project run at root:
`python hand_detection.py <optional_param>`

-   `-h` can be used to know what the available params are;
-   `--left` or `-l`: Moves the ROI to the left side;
-   `--shot` or `-s`: Detects hand in a single shot, a picture, not video;
-   `--input=<path>` or `-i=<path>`: Analyses a stored image;

Then the webcam image is shown saying 'Welcome'.

You can then:

-   Press `h` to go to the color calculator;
-   Or press `v` to go to the hand detection;

Press `q` at any time to close the project.

### Hand Contours

In this mode, you will calculate the color of the hand to be detected.

Place your hand in the _blue square_ and then

-   Press `v` to capture;

If the result pleases you

-   Press `any key` to proceed to video hand detection.
    If it doesn't:
-   Press `n` to return to the color calculator.

### Hand Detection

In this mode, you will be able to do signs with a single hand that will be analysed by the algorithm.

Check the list of available signs above.

To exit the app press `q`.

## Additional Information

For more information about the project consult the [wiki](https://github.com/BlueDi/Hand-Detection/wiki) of this project.

For more knowledge of the area we recommend the following works:

[[1]](http://hdl.handle.net/11025/11096) A. Birdal e R. Hassanpour, Region Based Hand Gesture Recognition. Václav Skala - UNION Agency, 2008.

[[2]](https://doi.org/10.1016/j.patcog.2015.07.014) Y. Zhou, G. Jiang, e Y. Lin, «A novel finger and hand pose estimation technique for real-time hand gesture recognition», Pattern Recognition, vol. 49, pp. 102–114, Jan. 2016.

[[3]](https://doi.org/10.1007/s10462-012-9356-9) S. S. Rautaray e A. Agrawal, «Vision based hand gesture recognition for human computer interaction: a survey», Artif Intell Rev, vol. 43, n. 1, pp. 1–54, Jan. 2015.

[[4]](http://hdl.handle.net/11025/1847) E. Sánchez-Nielsen, L. Antón-Canaĺıs, e M. Hernández-Tejera, «Hand Gesture Recognition for Human-Machine Interaction», em The 12-th International Conference in Central Europe on Computer Graphics, Visualization and Computer Vision’2004, WSCG 2004, University of West Bohemia, Campus Bory, Plzen-Bory, Czech Republic, February 2-6, 2004, 2004, pp. 395–402.
