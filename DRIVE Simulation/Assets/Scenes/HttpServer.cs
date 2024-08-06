using System;
using System.Net;
using System.Text;
using System.Threading;
using UnityEngine;

[System.Serializable]
public class ControlValues
{
    public float speed;
    public float angle;
}

public class HttpServer : MonoBehaviour
{
    private HttpListener listener;
    private Thread listenerThread;
    public int port = 8000;  // Portnummer, auf dem der Server lauschen wird
    public CustomCarController carController;

    void Start()
    {
        listener = new HttpListener();
        listener.Prefixes.Add($"http://*:{port}/");
        listener.Start();

        listenerThread = new Thread(StartListener);
        listenerThread.IsBackground = true;
        listenerThread.Start();

        Debug.Log("HTTP Server gestartet");
    }

    void OnApplicationQuit()
    {
        listener.Stop();
        listenerThread.Abort();
    }

    private void StartListener()
    {
        while (listener.IsListening)
        {
            try
            {
                var context = listener.GetContext();
                ProcessRequest(context);
            }
            catch (Exception ex)
            {
                Debug.Log($"HTTP Server Fehler: {ex.Message}");
            }
        }
    }

    private void ProcessRequest(HttpListenerContext context)
    {
        if (context.Request.HttpMethod == "POST")
        {
            using (var reader = new System.IO.StreamReader(context.Request.InputStream, context.Request.ContentEncoding))
            {
                var json = reader.ReadToEnd();
                var controlValues = JsonUtility.FromJson<ControlValues>(json);
                carController.SetControlValues(controlValues.speed, controlValues.angle);
                Debug.Log($"Empfangen: Speed={controlValues.speed}, Steering={controlValues.angle}");

                var responseString = "OK";
                var buffer = Encoding.UTF8.GetBytes(responseString);

                context.Response.ContentLength64 = buffer.Length;
                var responseOutput = context.Response.OutputStream;
                responseOutput.Write(buffer, 0, buffer.Length);
                responseOutput.Close();
            }
        }
    }
}
