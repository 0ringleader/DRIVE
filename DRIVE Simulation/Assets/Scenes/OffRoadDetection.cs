using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class OffRoadDetection : MonoBehaviour
{
    public string roadTag = "RoadMesh"; 
    private Dictionary<string, bool> wheelStatus; 
    public Text statusText; 
    public CustomCarController carController; // Reference to CustomCarController
    private RoadStatusSender roadStatusSender;

    void Start()
    {
        wheelStatus = new Dictionary<string, bool>
        {
            { "WheelHL", true },
            { "WheelHR", true },
            { "WheelVL", true },
            { "WheelVR", true }
        };
        statusText.gameObject.SetActive(false); 
        UpdateOffRoadStatus();
        roadStatusSender = FindObjectOfType<RoadStatusSender>();
    }

    void OnCollisionEnter(Collision collision)
    {
        Debug.Log("OnCollisionEnter aufgerufen mit Tag: " + collision.collider.tag);
        if (wheelStatus.ContainsKey(collision.collider.tag) || collision.collider.GetComponent<MeshCollider>() != null)
        {
            wheelStatus[collision.collider.tag] = true; 
            UpdateOffRoadStatus();
        }
    }

    void OnCollisionExit(Collision collision)
    {
        Debug.Log("OnCollisionExit aufgerufen mit Tag: " + collision.collider.tag);
        if (wheelStatus.ContainsKey(collision.collider.tag) || collision.collider.GetComponent<MeshCollider>() != null)
        {
            wheelStatus[collision.collider.tag] = false; 
            UpdateOffRoadStatus();
        }
    }

    void UpdateOffRoadStatus()
    {
        bool allOnRoad = true;
        bool allOffRoad = true;
        foreach (var status in wheelStatus.Values)
        {
            if (!status)
            {
                allOnRoad = false;
            }
            if (status)
            {
                allOffRoad = false;
            }
        }

        if (allOnRoad)
        {
            SetOffRoadStatus(false, null);
        }
        else if (allOffRoad)
        {
            SetOffRoadStatus(true, null);
        }
        else
        {
            foreach (var wheel in wheelStatus)
            {
                if (!wheel.Value)
                {
                    SetOffRoadStatus(true, wheel.Key);
                    return;
                }
            }
        }
    }

    void SetOffRoadStatus(bool offRoad, string wheelTag)
    {
        Debug.Log("SetOffRoadStatus aufgerufen: " + offRoad);
        statusText.gameObject.SetActive(true);
        if (offRoad)
        {
            if (wheelTag == null)
            {
                statusText.text = "Das Auto ist nicht auf der Straße.";
                statusText.color = Color.red;
                Debug.Log("Das Auto ist nicht auf der Straße.");
            }
            else
            {
                statusText.text = $"Das Rad {wheelTag} hat die Straße verlassen!";
                statusText.color = Color.red;
                Debug.Log($"Das Rad {wheelTag} hat die Straße verlassen!");
            }

            // Reset the car if resetOnOffRoad is true
            if (carController != null && carController.resetOnOffRoad)
            {
                carController.ResetCar();
            }
        }
        else
        {
            statusText.text = "Das Auto ist auf der Straße.";
            statusText.color = new Color(0.0f, 0.5f, 0.0f);
            Debug.Log("Das Auto ist auf der Straße.");
        }
        roadStatusSender.SendStatusToServer(offRoad);
    }
}