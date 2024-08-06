using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

[System.Serializable]
public class ControlValues2
{
    public float speed;
    public float steering;
}

public class dataReciever : MonoBehaviour
{
    public CustomCarController carController;
    private bool isRequesting = false;

    void Update()
    {
        if (!isRequesting)
        {
            StartCoroutine(RequestControlValues());
        }
    }

    IEnumerator RequestControlValues()
    {
        isRequesting = true;
        string url = "http://localhost:8000";
        using (UnityWebRequest webRequest = UnityWebRequest.Get(url))
        {
            yield return webRequest.SendWebRequest();

            if (webRequest.result == UnityWebRequest.Result.ConnectionError || webRequest.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.Log(webRequest.error);
            }
            else
            {
                ControlValues2 values = JsonUtility.FromJson<ControlValues2>(webRequest.downloadHandler.text);
                carController.SetControlValues(values.speed, values.steering); 
            }
        }
        isRequesting = false;
    }
}