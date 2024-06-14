using UnityEngine;

public class WheelControl : MonoBehaviour
{
    public Transform wheelModel;

    [HideInInspector] public WheelCollider WheelCollider;

    // Create properties for the CarControl script
    // (You should enable/disable these via the 
    // Editor Inspector window)
    public bool steerable;
    public bool motorized;

    Vector3 position;
    Quaternion rotation;

    // Start is called before the first frame update
    private void Start()
    {
        WheelCollider = GetComponent<WheelCollider>();
    }

    // Update is called once per frame
    void Update()
    {
        // Get the Wheel collider's world pose values
        WheelCollider.GetWorldPose(out position, out rotation);

        // Define the additional rotation to correct the wheel model's orientation
        Quaternion additionalRotation = Quaternion.Euler(0, 0, 270); // Adjust these values as needed

        // Apply the additional rotation
        rotation *= additionalRotation;

        // Use the corrected rotation to set the wheel model's position and rotation
        wheelModel.transform.position = position;
        wheelModel.transform.rotation = rotation;
    }
}