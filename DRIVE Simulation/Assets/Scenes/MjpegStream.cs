using System;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Collections;
using UnityEngine;

public class MJPEGStream : MonoBehaviour
{
    public int port = 8000;
    public Camera cameraToCapture;
    public int width = 640;
    public int height = 480;
    public float frameRate = 10f; // Frame rate in frames per second
    private float frameDuration; // Duration of one frame in milliseconds

    private HttpListener httpListener;
    private bool isStreaming;
    private RenderTexture renderTexture;
    private Texture2D screenshot;
    private readonly object frameLock = new object();
    private byte[] latestFrame = null;

    void Start()
    {
        // Ensure Unity continues running even when the game window is not in focus
        Application.runInBackground = true;

        // Initialize RenderTexture and Texture2D on the main thread
        renderTexture = new RenderTexture(width, height, 24);
        screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);

        // Compute the frame duration based on the desired frame rate
        frameDuration = 1000f / frameRate;

        StartServer();
        StartCoroutine(CaptureFrames());
    }

    void StartServer()
    {
        Debug.Log("Starting server...");
        httpListener = new HttpListener();
        httpListener.Prefixes.Add($"http://*:{port}/stream/");
        httpListener.Start();
        isStreaming = true;

        // Run the HttpListener in a separate thread
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

                if (jpegData == null)
                {
                    Thread.Sleep(10);
                    continue;
                }

                string header = "\r\n--boundary\r\nContent-Type: image/jpeg\r\nContent-Length: " + jpegData.Length + "\r\n\r\n";
                byte[] headerBytes = Encoding.ASCII.GetBytes(header);

                response.OutputStream.Write(headerBytes, 0, headerBytes.Length);
                response.OutputStream.Write(jpegData, 0, jpegData.Length);
                response.OutputStream.Flush();
                Debug.Log("Frame sent.");

                // Sleep for the frame duration to control the frame rate
                Thread.Sleep((int)frameDuration);
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

            cameraToCapture.targetTexture = renderTexture;
            cameraToCapture.Render();

            RenderTexture.active = renderTexture;
            screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            screenshot.Apply();

            byte[] bytes = screenshot.EncodeToJPG();

            cameraToCapture.targetTexture = null;
            RenderTexture.active = null;

            lock (frameLock)
            {
                latestFrame = bytes;
            }
        }
    }

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
}
