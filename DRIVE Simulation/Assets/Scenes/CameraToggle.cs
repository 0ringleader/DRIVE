using UnityEngine;

public class CameraToggle : MonoBehaviour
{
    // Variable zum Verfolgen des Kamera-Rendering-Zustands
    private bool isCameraEnabled = true;

    void Update()
    {
        // Überprüfen, ob die C-Taste gedrückt wurde
        if (Input.GetKeyDown(KeyCode.C))
        {
            ToggleCameraRendering();
        }
    }

    void ToggleCameraRendering()
    {
        // Kamera-Rendering ein- oder ausschalten
        isCameraEnabled = !isCameraEnabled;
        GetComponent<Camera>().enabled = isCameraEnabled;
    }
}