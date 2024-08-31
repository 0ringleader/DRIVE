using System;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Collections;
using UnityEngine;

public class MJPEGStream : MonoBehaviour
{
    // Stream configuration
    public int port = 8000;
    public string subdir = "stream";
    public Camera cameraToCapture;
    public int width = 300;
    public int height = 225;
    public float frameRate = 10f; // Frame rate in frames per second
    private float frameDuration; // Duration of one frame in milliseconds
    public bool rotateImage = true; // Determines if the image should be rotated
    public bool syncToGameSpeed = false; // Sync frame rate to the simulation's speed
    private float gameSpeed = 1f;


    private HttpListener httpListener;
    private bool isStreaming;
    private RenderTexture renderTexture;
    private Texture2D screenshot;
    private readonly object frameLock = new object(); // Lock object for thread safety
    private byte[] latestFrame = null; // Latest frame to be sent to clients

    void Start()
    {
        // Ensure Unity continues running even when the game window is not in focus
        Application.runInBackground = true;

        // Initialize RenderTexture and Texture2D
        renderTexture = new RenderTexture(width, height, 24);
        screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);

        // Compute the frame duration based on the desired frame rate
        frameDuration = 1000f / frameRate;

        // Start the server and begin capturing frames
        StartServer();
        StartCoroutine(CaptureFrames());
    }

    void StartServer()
    {
        Debug.Log("Starting server...");
        httpListener = new HttpListener();
        httpListener.Prefixes.Add($"http://*:{port}/{subdir}/");
        httpListener.Start();
        isStreaming = true;

        // Run the HttpListener in a separate thread to not interrupt Unity's main thread
        // I had a lot of freezes when running the HttpListener in the main thread
        new Thread(() =>
        {
            while (isStreaming)
            {
                Debug.Log("Waiting for client connection...");
                try
                {
                    var context = httpListener.GetContext();
                    ThreadPool.QueueUserWorkItem(o => HandleClient(context));
                }
                catch (Exception e)
                {
                    Debug.LogError($"Error accepting clients: {e.Message}");
                }
            }
        }).Start();

        Debug.Log($"MJPEG stream available at http://localhost:{port}/stream/");
    }

    void HandleClient(HttpListenerContext context)
    {
        var response = context.Response;
        response.ContentType = "multipart/x-mixed-replace; boundary=--boundary";
        response.StatusCode = (int)HttpStatusCode.OK;

        Debug.Log("Client connected.");

        try
        {
            while (isStreaming)
            {
                byte[] jpegData;
                lock (frameLock)
                {
                    jpegData = latestFrame;
                }

                // If no frame is ready, wait briefly before retrying
                if (jpegData == null)
                {
                    Thread.Sleep(10);
                    continue;
                }

                // Write the JPEG frame to the response stream
                string header = "\r\n--boundary\r\nContent-Type: image/jpeg\r\nContent-Length: " + jpegData.Length + "\r\n\r\n";
                byte[] headerBytes = Encoding.ASCII.GetBytes(header);

                response.OutputStream.Write(headerBytes, 0, headerBytes.Length);
                response.OutputStream.Write(jpegData, 0, jpegData.Length);
                response.OutputStream.Flush();
                Debug.Log("Frame sent.");

                // Control the frame rate based on simulations speed to send more or less data to neural networks accordingly
                if (syncToGameSpeed)
                {
                    Thread.Sleep((int)(frameDuration / gameSpeed));
                }
                else
                {
                    Thread.Sleep((int)frameDuration);
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error sending frame: {ex.Message}");
        }
        finally
        {
            response.OutputStream.Close();
            Debug.Log("Client disconnected.");
        }
    }

    IEnumerator CaptureFrames()
    {
        while (isStreaming)
        {
            yield return new WaitForEndOfFrame();

            // Capture the frame from the camera
            cameraToCapture.targetTexture = renderTexture;
            cameraToCapture.Render();

            RenderTexture.active = renderTexture;
            screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            screenshot.Apply();

            // Optionally rotate the captured image (real camera is mounted upside down)
            if (rotateImage)
            {
                RotateTexture(screenshot);
            }
            byte[] bytes = screenshot.EncodeToJPG();

            cameraToCapture.targetTexture = null;
            RenderTexture.active = null;

            // Store the frame in a thread-safe manner
            lock (frameLock)
            {
                latestFrame = bytes;
            }
        }
    }

    // Rotates the captured Texture2D image by 180 degrees
    void RotateTexture(Texture2D texture)
    {
        Color[] pixels = texture.GetPixels();
        Color[] rotatedPixels = new Color[pixels.Length];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                rotatedPixels[(height - 1 - y) * width + (width - 1 - x)] = pixels[y * width + x];
            }
        }

        texture.SetPixels(rotatedPixels);
        texture.Apply();
    }

    // Stops the server when the application is closed
    void OnApplicationQuit()
    {
        Debug.Log("Stopping server...");
        isStreaming = false;
        if (httpListener != null && httpListener.IsListening)
        {
            httpListener.Stop();
            httpListener.Close();
        }
    }

    // Adjusts the simulation speed, affecting the frame rate if syncToGameSpeed is true
    public void setGameSpeed(float speed)
    {
        gameSpeed = speed;
    }
}
