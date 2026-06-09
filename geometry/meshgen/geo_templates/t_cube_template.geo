SetFactory("OpenCASCADE");

// Characteristic mesh length
h = DEFINE_H;
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;

// Cylinder dimensions
LOWER_LENGTH = 30.0 / 100.0;
LOWER_RADIUS = 4.5 / 100.0;
UPPER_LENGTH = 30.0 / 100.0;
UPPER_RADIUS = 4.5 / 100.0;
MIDDLE_LENGTH = 120.0 / 100.0;
MIDDLE_RADIUS = 6.5 / 100.0;

// Offset for positioning along the X-axis
OFFSET = 0.0;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// First Cylinder - Lower /////////////////////////////////////////////
////////////////////////////////////////////// ID - 00001 ////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// Third Cylinder - Middle /////////////////////////////////////////////
////////////////////////////////////////////// ID - 10001 ////////////////////////////////////////////////////////

// Create a box domain that is used to slice potential "tilted" bottom part of the rotated conduit - we want it to be
//  parallel with the ZY plane
Box(2) = {-MIDDLE_LENGTH / 2.0, -MIDDLE_RADIUS, LOWER_LENGTH - MIDDLE_RADIUS, MIDDLE_LENGTH, 2*MIDDLE_RADIUS, 2*MIDDLE_RADIUS};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// Union and Cleanup ///////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Union the three volumes and delete original parts
BooleanUnion{ Volume{1}; Delete; }{ Volume{2}; Delete; }

// Delete any remaining surfaces
Recursive Delete {
  Surface{1};
}

// Delete any remaining lines
Recursive Delete {
  Line{1};
}

