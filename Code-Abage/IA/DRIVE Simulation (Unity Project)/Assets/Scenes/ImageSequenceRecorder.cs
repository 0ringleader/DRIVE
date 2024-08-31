using System.Collections;
using System.IO;
using UnityEngine;

public class ImageSequenceRecorder : MonoBehaviour
{
    public Camera cameraToCapture;
    public int width = 640;
    public int height = 480;
    public string folderPath = "ImageSequence";
    public bool isRecording = false;
    public float frameRate = 10f; // Desired frame rate

    private RenderTexture renderTexture;
    private Texture2D screenshot;
    private int frameCount = 0;
    private float frameDuration; // Duration of one frame in seconds
    private float nextFrameTime; // Time when the next frame should be captured

    void Start()
    {
        // Initialize RenderTexture and Texture2D
        renderTexture = new RenderTexture(width, height, 24);
        screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);

        // Ensure the folder exists
        if (!Directory.Exists(folderPath))
        {
            Directory.CreateDirectory(folderPath);
        }

        // Compute the frame duration based on the desired frame rate
        frameDuration = 1f / frameRate;
    }

    void Update()
    {
        if (isRecording)
        {
            // Check if it's time to capture the next frame
            if (Time.time >= nextFrameTime)
            {
                StartCoroutine(CaptureAndSaveScreenshot());
                nextFrameTime = Time.time + frameDuration; // Schedule the next frame capture
            }
        }
    }

    IEnumerator CaptureAndSaveScreenshot()
    {
        // Capture the screenshot
        cameraToCapture.targetTexture = renderTexture;
        cameraToCapture.Render();

        RenderTexture.active = renderTexture;
        screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        screenshot.Apply();

        byte[] bytes = screenshot.EncodeToJPG();

        cameraToCapture.targetTexture = null;
        RenderTexture.active = null;

        // Save the screenshot to the specified folder
        string filePath = Path.Combine(folderPath, $"frame_{frameCount:D04}.jpg");
        File.WriteAllBytes(filePath, bytes);

        Debug.Log($"Saved frame {frameCount} to {filePath}");

        frameCount++;

        // Yielding here to avoid blocking the main thread, but still controlling the frame capture rate
        yield return null;
    }

    void OnApplicationQuit()
    {
        // Clean up
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
        }
        if (screenshot != null)
        {
            Destroy(screenshot);
        }
    }
}
