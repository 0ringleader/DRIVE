using UnityEngine;

public class CameraToggle : MonoBehaviour
{
    // Variable to track the camera rendering state
    private bool isCameraEnabled = true;

    void Update()
    {
        // Check if the 'C' key was pressed
        if (Input.GetKeyDown(KeyCode.C))
        {
            ToggleCameraRendering();
        }
    }

    void ToggleCameraRendering()
    {
        // Toggle rendering from the camera on or off (excluding the rendering for the stream)
        isCameraEnabled = !isCameraEnabled;
        GetComponent<Camera>().enabled = isCameraEnabled;
    }
}