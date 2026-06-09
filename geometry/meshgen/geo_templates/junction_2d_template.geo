SetFactory("OpenCASCADE");

// 1 optimisation parameter - offset of the bottom cylinder
// upper inflow should be 0.4 ms-1
// lower inflow should be 0.5 ms-1

// Characteristic mesh length
h = 0.001;  // Suggested characteristic mesh length for balanced detail and computation cost
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;

// Offset for positioning along the X-axis - bounds -0.02 to +0.02
// OFFSET = 0.00;  // Adjust if needed for positioning; set to 0 for centered positioning
OFFSET = DEFINE_OFFSET;

// Rotation of the conduit - bounds -0.5 to +0.5
LOWER_ANGLE = DEFINE_LOWER_ANGLE;
LOWER_ANGLE = (3.14159 / 180) * LOWER_ANGLE;

// Rotation of the vena cava - bounds -0.5 to +0.5
UPPER_ANGLE = DEFINE_UPPER_ANGLE;
UPPER_ANGLE = (3.14159 / 180) * UPPER_ANGLE;

// Modification of the width of the conduit - bounds -0.001 to +0.001
EXTRA_WIDTH = 0.000;
// EXTRA_WIDTH = DEFINE_EXTRA_WIDTH

// Flaring of the connections - bounds 0.0 to 0.0035
UPPER_FLARE = DEFINE_UPPER_FLARE;
LOWER_FLARE = DEFINE_LOWER_FLARE;

// Cylinder dimensions
LOWER_LENGTH = 0.06;  // Length of the lower (bottom) cylinder in meters
LOWER_RADIUS = 0.007 + EXTRA_WIDTH; // Radius of the lower cylinder in meters
UPPER_LENGTH = 0.06;  // Length of the upper (top) cylinder in meters
UPPER_RADIUS = 0.007; // Radius of the upper cylinder in meters
MIDDLE_LENGTH = 0.17; // Length of the middle (horizontal) cylinder in meters
MIDDLE_RADIUS = 0.011;  // Radius of the middle cylinder in meters

RESOLUTION = DEFINE_RESOLUTION;
If (RESOLUTION <= 0)
  RESOLUTION = 1;
EndIf
PAD = MIDDLE_LENGTH / (128 * RESOLUTION * 8.0);

CYLINDER_SHIFT = 0.002;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// First Cylinder - Lower /////////////////////////////////////////////
////////////////////////////////////////////// ID - 00001 ////////////////////////////////////////////////////////

/*
// Define points along the axis of the lower cylinder
Point(1) = {OFFSET, 0.0, LOWER_LENGTH, h};
Point(2) = {OFFSET, 0.0, 0.0, h};

// Create line and wire for lower cylinder extrusion
Line(1) = {2, 1};
Wire(2) = {1};

// Disk representing the base of the lower cylinder
Disk(1) = {OFFSET, 0.0, LOWER_LENGTH, LOWER_RADIUS};

// Extrude the surface to form the first cylinder volume
Extrude { Surface{1}; } Using Wire {2}
*/

// Points that determine the central axis of the conduit - used for forming a closed loop
Point(1) = {OFFSET, 0.0, 0.0, h};
Point(11) = {OFFSET, 0.0, LOWER_LENGTH + CYLINDER_SHIFT, h};

// Main points forming the spline to be rotated and revolved
Point(2) = {OFFSET + LOWER_RADIUS, 0.0, 0.0, h};
Point(3) = {OFFSET + LOWER_RADIUS, 0.0, 1.0 * LOWER_LENGTH / 8.0, h};
Point(4) = {OFFSET + LOWER_RADIUS, 0.0, 2.0 * LOWER_LENGTH / 8.0, h};
Point(5) = {OFFSET + LOWER_RADIUS, 0.0, 3.0 * LOWER_LENGTH / 8.0, h};
Point(6) = {OFFSET + LOWER_RADIUS, 0.0, LOWER_LENGTH / 2.0, h};
Point(7) = {OFFSET + LOWER_RADIUS + LOWER_FLARE / 4.0, 0.0, 5.0 * LOWER_LENGTH / 8.0, h};
Point(8) = {OFFSET + LOWER_RADIUS + 2.5 * LOWER_FLARE / 4.0, 0.0, 6.0 * LOWER_LENGTH / 8.0, h};
Point(9) = {OFFSET + LOWER_RADIUS + 3.0 * LOWER_FLARE / 4.0, 0.0, 7.0 * LOWER_LENGTH / 8.0, h};
Point(10) = {OFFSET + LOWER_RADIUS + LOWER_FLARE, 0.0, LOWER_LENGTH + CYLINDER_SHIFT, h};

// Lines that form a close loop
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(4) = {10, 11};
Line(5) = {11, 1};

// Spline connecting the main points
Spline(3) = {3, 4, 5, 6, 7, 8, 9, 10};

// Closed loop formed by the lines
Curve Loop(1) = {1, 2, 3, 4, 5};

// Make a surface inside of the closed loop
Plane Surface(1) = {1};

// Revolve the surface around the axis X that is translated to the central point of the conduit creating Volume{1}
Extrude { {0, 0, 1}, {OFFSET, 0.0, 0.00} , 2*Pi } {
  Surface{1}; Recombine;
}

// Rotate the created volume by specified angle around the axis Z that is translated to the central point of the conduit
Rotate { {0, 1, 0}, {OFFSET, 0.0, LOWER_LENGTH + CYLINDER_SHIFT}, LOWER_ANGLE} {
  Volume{1};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// Second Cylinder - Upper ////////////////////////////////////////////
////////////////////////////////////////////// ID - 00101 ////////////////////////////////////////////////////////

// Define points along the axis of the upper cylinder
Point(111) = {0.0, 0.0, LOWER_LENGTH - CYLINDER_SHIFT, h};
Point(101) = {0.0, 0.0, LOWER_LENGTH - CYLINDER_SHIFT + UPPER_LENGTH, h};

// Main points forming the spline to be rotated and revolved
Point(102) = {UPPER_RADIUS, 0.0, LOWER_LENGTH - CYLINDER_SHIFT + UPPER_LENGTH, h};
Point(103) = {UPPER_RADIUS, 0.0, LOWER_LENGTH - CYLINDER_SHIFT + UPPER_LENGTH - 1.0 * UPPER_LENGTH / 8.0, h};
Point(104) = {UPPER_RADIUS, 0.0, LOWER_LENGTH - CYLINDER_SHIFT + UPPER_LENGTH - 2.0 * UPPER_LENGTH / 8.0, h};
Point(105) = {UPPER_RADIUS, 0.0, LOWER_LENGTH - CYLINDER_SHIFT + UPPER_LENGTH - 3.0 * UPPER_LENGTH / 8.0, h};
Point(106) = {UPPER_RADIUS, 0.0, LOWER_LENGTH - CYLINDER_SHIFT + UPPER_LENGTH - UPPER_LENGTH / 2.0, h};
Point(107) = {UPPER_RADIUS + UPPER_FLARE / 4.0, 0.0, LOWER_LENGTH - CYLINDER_SHIFT + UPPER_LENGTH - 5.0 * UPPER_LENGTH / 8.0, h};
Point(108) = {UPPER_RADIUS + 2.5 * UPPER_FLARE / 4.0, 0.0, LOWER_LENGTH - CYLINDER_SHIFT + UPPER_LENGTH - 6.0 * UPPER_LENGTH / 8.0, h};
Point(109) = {UPPER_RADIUS + 3.0 * UPPER_FLARE / 4.0, 0.0, LOWER_LENGTH - CYLINDER_SHIFT + UPPER_LENGTH - 7.0 * UPPER_LENGTH / 8.0, h};
Point(110) = {UPPER_RADIUS + UPPER_FLARE, 0.0, UPPER_LENGTH - CYLINDER_SHIFT, h};

// Lines that form a close loop
Line(101) = {101, 102};
Line(102) = {102, 103};
Line(104) = {110, 111};
Line(105) = {111, 101};

// Spline connecting the main points
Spline(103) = {103, 104, 105, 106, 107, 108, 109, 110};

// Closed loop formed by the lines
Curve Loop(101) = {101, 102, 103, 104, 105};

// Make a surface inside of the closed loop
Plane Surface(101) = {101};

// Revolve the surface around the axis X that is translated to the central point of the conduit creating Volume{1}
Extrude { {0, 0, 1}, {0.0, 0.0, LOWER_LENGTH - CYLINDER_SHIFT + UPPER_LENGTH} , 2*Pi } {
  Surface{101}; Recombine;
}

// Rotate the created volume by specified angle around the axis Z that is translated to the central point of the conduit
Rotate { {0, 1, 0}, {0.0, 0.0, LOWER_LENGTH - CYLINDER_SHIFT}, UPPER_ANGLE} {
  Volume{2};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// Third Cylinder - Middle /////////////////////////////////////////////
////////////////////////////////////////////// ID - 10001 ////////////////////////////////////////////////////////

// Define points along the axis of the middle cylinder
Point(10001) = {-MIDDLE_LENGTH / 2.0, 0.0, LOWER_LENGTH, h};
Point(10002) = {MIDDLE_LENGTH / 2.0 + PAD, 0.0, LOWER_LENGTH, h};

// Create line and wire for middle cylinder extrusion
Line(10001) = {10002, 10001};
Wire(10002) = {10001};

// Disk representing the base of the middle cylinder
Disk(10001) = {-MIDDLE_LENGTH / 2.0, 0.0, LOWER_LENGTH, MIDDLE_RADIUS};

// Orient disk along the XY plane
Rotate{{0, 1, 0}, {-MIDDLE_LENGTH / 2.0, 0.0, LOWER_LENGTH}, Pi/2}{Surface{10001};}

// Extrude the surface to form the third cylinder volume
Extrude { Surface{10001}; } Using Wire {10002}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// Union and Cleanup ///////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Union the three volumes and delete original parts
BooleanUnion(4) = { Volume{1}; Delete; }{ Volume{2, 3}; Delete; };

// Delete any remaining surfaces
Recursive Delete {
  Surface{1, 101, 10001};
}

// Delete any remaining lines
Recursive Delete {
  Line{10001};
}

Recursive Delete {
  Point{4, 5, 6, 7, 8, 9, 104, 105, 106, 107, 108, 109};
}

// Create a box domain that is used to slice potential "tilted" bottom part of the rotated conduit - we want it to be
//  parallel with the ZY plane
Box(5) = {-MIDDLE_LENGTH / 2.0, -MIDDLE_RADIUS, 0.015, MIDDLE_LENGTH + PAD, 2*MIDDLE_RADIUS, LOWER_LENGTH + UPPER_LENGTH - 0.03};

// Intersect the box with the Volume {4}
BooleanIntersection(7) = { Volume{5}; Delete;}{ Volume{4}; Delete;};
