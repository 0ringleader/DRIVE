using System;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Collections;
using UnityEngine;
using Object = UnityEngine.Object;

public class MJPEGStreamCodeTest : MonoBehaviour
{
    public int port = 8000;
    public Camera cameraToCapture;
    public int width = 640;
    public int height = 480;
    private HttpListener httpListener;
    private bool isStreaming;
    private RenderTexture renderTexture;
    private Texture2D screenshot;

    void Start()
    {
        // Initialize RenderTexture and Texture2D on the main thread
        renderTexture = new RenderTexture(width, height, 24);
        screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);

        StartServer();
    }

    void StartServer()
    {
        Debug.Log("Starting server...");
        httpListener = new HttpListener();
        httpListener.Prefixes.Add($"http://*:{port}/stream/");
        httpListener.Start();
        isStreaming = true;
        StartCoroutine(StreamMJPEG());
        Debug.Log($"MJPEG stream available at http://localhost:{port}/stream/");
    }

    IEnumerator StreamMJPEG()
    {
        while (isStreaming)
        {
            Debug.Log("Waiting for client connection...");
            var context = httpListener.GetContext();
            var response = context.Response;
            response.ContentType = "multipart/x-mixed-replace; boundary=--boundary";
            response.StatusCode = (int)HttpStatusCode.OK;

            Debug.Log("Client connected.");

            while (isStreaming)
            {
                byte[] jpegData = CaptureScreenshot();

                string header = "\r\n--boundary\r\nContent-Type: image/jpeg\r\nContent-Length: " + jpegData.Length + "\r\n\r\n";
                byte[] headerBytes = Encoding.ASCII.GetBytes(header);

                try
                {
                    response.OutputStream.Write(headerBytes, 0, headerBytes.Length);
                    response.OutputStream.Write(jpegData, 0, jpegData.Length);
                    response.OutputStream.Flush();
                    Debug.Log("Frame sent.");
                }
                catch (Exception ex)
                {
                    Debug.Log($"Error sending frame: {ex.Message}");
                    break;
                }

                yield return new WaitForSeconds(0.1f); // Adjust the frame rate as needed
            }

            response.OutputStream.Close();
            Debug.Log("Client disconnected.");
        }
    }

    byte[] CaptureScreenshot()
    {
        cameraToCapture.targetTexture = renderTexture;
        cameraToCapture.Render();

        RenderTexture.active = renderTexture;
        screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        screenshot.Apply();

        byte[] bytes = screenshot.EncodeToJPG();

        cameraToCapture.targetTexture = null;
        RenderTexture.active = null;

        return bytes;
    }

    void OnApplicationQuit()
    {
        Debug.Log("Stopping server...");
        isStreaming = false;
        httpListener.Stop();
        httpListener.Close();
    }
}
