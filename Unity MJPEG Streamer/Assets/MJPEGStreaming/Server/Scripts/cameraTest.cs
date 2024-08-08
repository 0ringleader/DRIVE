using System.IO;
using UnityEngine;

public class cameraTest : MonoBehaviour
{
    public Camera cameraToCapture;
    public string filePath = "screenshot.jpg";
    public int width = 1920;
    public int height = 1080;
    public KeyCode captureKey = KeyCode.Space; // Set the key to capture screenshot

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.P))
        {
            CaptureScreenshot();
        }
    }

    public void CaptureScreenshot()
    {
        RenderTexture renderTexture = new RenderTexture(width, height, 24);
        cameraToCapture.targetTexture = renderTexture;
        Texture2D screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);

        cameraToCapture.Render();

        RenderTexture.active = renderTexture;
        screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        screenshot.Apply();

        byte[] bytes = screenshot.EncodeToJPG();
        string fullPath = Path.Combine(Application.dataPath, filePath);
        File.WriteAllBytes(fullPath, bytes);

        cameraToCapture.targetTexture = null;
        RenderTexture.active = null;
        Destroy(renderTexture);
        Destroy(screenshot);

        Debug.Log($"Screenshot saved to {fullPath}");
    }
}