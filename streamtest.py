using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Threading;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class MJPEGStreaming : MonoBehaviour
{
    public int port = 8080; // Port, über den der Stream bereitgestellt wird
    public int quality = 50; // Bildqualität (zwischen 1 und 100)
    public RawImage imageDisplay; // RawImage zum Anzeigen des Streams in Unity
    public Camera streamingCamera; // Die Kamera, deren Bild gestreamt werden soll

    private Texture2D texture;
    private Thread serverThread;
    private bool isStreaming = false;

    void Start()
    {
        texture = new Texture2D(streamingCamera.targetTexture.width, streamingCamera.targetTexture.height, TextureFormat.RGB24, false);
        imageDisplay.texture = texture;

        serverThread = new Thread(new ThreadStart(ServerThread));
        serverThread.Start();
    }

    void OnDestroy()
    {
        isStreaming = false;
        serverThread.Abort();
    }

    void ServerThread()
    {
        TcpListener listener = new TcpListener(IPAddress.Any, port);
        listener.Start();

        isStreaming = true;
        while (isStreaming)
        {
            using (TcpClient client = listener.AcceptTcpClient())
            {
                using (NetworkStream stream = client.GetStream())
                {
                    byte[] header = Encoding.ASCII.GetBytes("HTTP/1.0 200 OK\r\n" +
                                                           "Connection: close\r\n" +
                                                           "Max-Age: 0\r\n" +
                                                           "Expires: 0\r\n" +
                                                           "Cache-Control: no-cache, private\r\n" +
                                                           "Pragma: no-cache\r\n" +
                                                           "Content-Type: multipart/x-mixed-replace; boundary=myboundary\r\n\r\n");

                    stream.Write(header, 0, header.Length);

                    while (isStreaming)
                    {
                        byte[] jpg = texture.EncodeToJPG(quality);
                        byte[] boundary = Encoding.ASCII.GetBytes("\r\n--myboundary\r\n" +
                                                                  "Content-Type: image/jpeg\r\n" +
                                                                  "Content-Length: " + jpg.Length + "\r\n\r\n");

                        stream.Write(boundary, 0, boundary.Length);
                        stream.Write(jpg, 0, jpg.Length);
                        stream.Flush();

                        Thread.Sleep(1000 / 30); // Stream mit ca. 30 FPS
                    }
                }
            }
        }
        listener.Stop();
    }

    void Update()
    {
        RenderTexture.active = streamingCamera.targetTexture;
        texture.ReadPixels(new Rect(0, 0, streamingCamera.targetTexture.width, streamingCamera.targetTexture.height), 0, 0);
        texture.Apply();
        RenderTexture.active = null;
    }
}
