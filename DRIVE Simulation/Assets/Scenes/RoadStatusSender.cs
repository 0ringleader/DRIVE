using System;
using System.Net;
using System.Text;
using System.Threading;
using UnityEngine;

[System.Serializable]
public class OffRoadStatus
{
    public bool isOffRoad;
}

public class RoadStatusSender : MonoBehaviour
{
    private HttpListener listener;
    private Thread listenerThread;
    public int port = 8000;  // Portnummer, auf dem der Server lauschen wird

    void Start()
    {
        listener = new HttpListener();
        listener.Prefixes.Add($"http://*:{port}/RoadStatusDetection");
        listener.Start();

        listenerThread = new Thread(StartListener);
        listenerThread.IsBackground = true;
        listenerThread.Start();

        Debug.Log("HTTP Status Server gestartet");
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
                Debug.Log($"HTTP Status Server Fehler: {ex.Message}");
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
                var status = JsonUtility.FromJson<OffRoadStatus>(json);
                Debug.Log($"Empfangen: isOffRoad={status.isOffRoad}");

                var responseString = "OK";
                var buffer = Encoding.UTF8.GetBytes(responseString);

                context.Response.ContentLength64 = buffer.Length;
                var responseOutput = context.Response.OutputStream;
                responseOutput.Write(buffer, 0, buffer.Length);
                responseOutput.Close();
            }
        }
    }

    public async void SendStatusToServer(bool isOffRoad)
    {
        string url = $"http://localhost:{port}/"; // Ersetze dies durch deine Server-URL
        var json = JsonUtility.ToJson(new OffRoadStatus { isOffRoad = isOffRoad });

        var request = (HttpWebRequest)WebRequest.Create(url);
        request.Method = "POST";
        request.ContentType = "application/json";
        var buffer = Encoding.UTF8.GetBytes(json);
        request.ContentLength = buffer.Length;

        using (var stream = request.GetRequestStream())
        {
            stream.Write(buffer, 0, buffer.Length);
        }

        try
        {
            var response = (HttpWebResponse)await request.GetResponseAsync();
            if (response.StatusCode == HttpStatusCode.OK)
            {
                Debug.Log("Status erfolgreich gesendet.");
            }
            else
            {
                Debug.Log("Fehler beim Senden des Status: " + response.StatusDescription);
            }
        }
        catch (Exception ex)
        {
            Debug.Log("Fehler beim Senden des Status: " + ex.Message);
        }
    }
}
