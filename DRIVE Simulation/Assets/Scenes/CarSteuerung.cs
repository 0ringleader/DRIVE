using UnityEngine;

public class CarSteuerung : MonoBehaviour
{
    public float speed = 10.0f;
    public float steering = 0.0f;
    public float turnSpeed = 5.0f; // Geschwindigkeit der Lenkung
    private Transform frontWheelLeft;
    private Transform frontWheelRight;
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        frontWheelLeft = transform.Find("VL");
        frontWheelRight = transform.Find("VR");
    }

    void Update()
    {
        // Get the input from the arrow keys and the mouse wheel
        steering = Mathf.Clamp(Input.GetAxis("Horizontal") * 35, -35, 35);
        speed += Input.GetAxis("Mouse ScrollWheel");

        // Rotate the front wheels based on the steering input
        frontWheelLeft.localEulerAngles = new Vector3(0, steering, 0);
        frontWheelRight.localEulerAngles = new Vector3(0, steering, 0);
    }

    void FixedUpdate()
    {
        // Rotate the forward vector 90 degrees around the Y axis
        Vector3 sideways = Quaternion.Euler(0, -90, 0) * transform.forward;

        // Move the car sideways
        Vector3 movement = sideways * speed * Time.fixedDeltaTime;
        rb.MovePosition(rb.position + movement);

        // Turn the car
        if (rb.velocity.magnitude > 0.1f)
        {
            float turn = steering * Time.fixedDeltaTime;
            Quaternion targetRotation = Quaternion.Euler(0, turn, 0);
            rb.rotation = Quaternion.Lerp(rb.rotation, targetRotation, turnSpeed * Time.fixedDeltaTime);
        }
    }
}