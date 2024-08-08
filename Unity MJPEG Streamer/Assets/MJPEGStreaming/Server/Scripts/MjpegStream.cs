using System;
using System.Collections;
using System.IO;
using System.Net;
using System.Text;
using UnityEngine;
using Object = UnityEngine.Object;

public class MjpegStream : MonoBehaviour
{
    public int port = 8000;
    private HttpListener httpListener;
    private bool isStreaming;

    void Start()
    {
        StartServer();
    }

    void StartServer()
    {
        httpListener = new HttpListener();
        httpListener.Prefixes.Add($"http://*:{port}/stream/");
        httpListener.Start();
        isStreaming = true;
        httpListener.BeginGetContext(OnContextReceived, null);
        Debug.Log($"MJPEG stream available at http://localhost:{port}/stream/");
    }

    void OnContextReceived(IAsyncResult result)
    {
        if (!isStreaming) return;

        var context = httpListener.EndGetContext(result);
        httpListener.BeginGetContext(OnContextReceived, null);

        var response = context.Response;
        response.ContentType = "multipart/x-mixed-replace; boundary=--frame";
        response.StatusCode = (int)HttpStatusCode.OK;

        StartCoroutine(StreamFrames(response));
    }

    IEnumerator StreamFrames(HttpListenerResponse response)
    {
        while (isStreaming)
        {
            Texture2D screenshot = ScreenCapture.CaptureScreenshotAsTexture();
            byte[] jpegData = screenshot.EncodeToJPG();
            Object.Destroy(screenshot);

            string header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + jpegData.Length + "\r\n\r\n";
            byte[] headerBytes = Encoding.ASCII.GetBytes(header);

            try
            {
                response.OutputStream.Write(headerBytes, 0, headerBytes.Length);
                response.OutputStream.Write(jpegData, 0, jpegData.Length);
                response.OutputStream.Write(Encoding.ASCII.GetBytes("\r\n"), 0, 2);
                response.OutputStream.Flush();
            }
            catch
            {
                break;
            }

            yield return new WaitForSeconds(0.1f); // Adjust the frame rate as needed
        }

        response.OutputStream.Close();
    }

    void OnApplicationQuit()
    {
        isStreaming = false;
        httpListener.Stop();
        httpListener.Close();
    }
}