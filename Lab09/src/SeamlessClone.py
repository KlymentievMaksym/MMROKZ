import cv2
import numpy as np
from scipy.sparse import lil_matrix, linalg

class PoissonEditor:
    def __init__(self, src, dst, mask):
        self.src = src
        self.dst = dst
        self.mask = mask
        
        # 1. Binarize mask (0 or 1) and normalize
        _, self.mask = cv2.threshold(self.mask, 127, 1, cv2.THRESH_BINARY)
        self.mask = self.mask.astype(np.uint8)

        # 2. Get coordinates of all white pixels in the mask
        # shape: (N, 2) -> (y, x)
        self.coords = np.argwhere(self.mask == 1)
        
        # 3. Create a mapping: Pixel Coordinate (y,x) -> Variable Index (0..N)
        # This helps us build the matrix A efficiently
        self.index_map = np.zeros(self.mask.shape, dtype=np.int32) - 1
        for i, (y, x) in enumerate(self.coords):
            self.index_map[y, x] = i
            
        self.N = len(self.coords) # Number of unknown pixels

    def get_laplacian_matrix(self):
        """
        Builds the Matrix A (Sparse).
        Diagonal is 4. Neighbors are -1.
        """
        A = lil_matrix((self.N, self.N))
        
        # Offsets for 4 neighbors: Up, Down, Left, Right
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i, (y, x) in enumerate(self.coords):
            A[i, i] = 4  # Center pixel coefficient
            
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                
                # If neighbor is inside the mask, it's a variable in our system
                if self.mask[ny, nx] == 1:
                    j = self.index_map[ny, nx]
                    A[i, j] = -1
        
        return A

    def get_b_vector(self, channel_src, channel_dst):
        """
        Calculates vector b.
        b = Laplacian(Source) + Boundary(Target)
        """
        b = np.zeros(self.N)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i, (y, x) in enumerate(self.coords):
            # 1. Laplacian of Source Image (Guidance field)
            # source_val * 4 - neighbors
            laplacian_val = 4 * channel_src[y, x] - \
                            float(channel_src[y-1, x]) - \
                            float(channel_src[y+1, x]) - \
                            float(channel_src[y, x-1]) - \
                            float(channel_src[y, x+1])
            
            b[i] = laplacian_val

            # 2. Add Boundary Conditions (Dirichlet)
            # If a neighbor is OUTSIDE the mask, we add the Target image value
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if self.mask[ny, nx] == 0:
                    b[i] += channel_dst[ny, nx]
        
        return b

    def blend(self):
        print(f"Solving Poisson Equation for {self.N} pixels...")
        
        # 1. Build Matrix A (Same for all channels)
        A = self.get_laplacian_matrix()
        A_csr = A.tocsr() # Convert to Compressed Sparse Row for speed

        # Prepare result image
        result = self.dst.copy()

        # 2. Solve for each channel (B, G, R) independently
        for channel in range(3):
            print(f"  > Processing Channel {channel}...")
            
            # Get source and destination intensity for this channel
            s_ch = self.src[:, :, channel].astype(float)
            d_ch = self.dst[:, :, channel].astype(float)
            
            # Build b vector
            b = self.get_b_vector(s_ch, d_ch)
            
            # Solve Ax = b
            x = linalg.spsolve(A_csr, b)
            
            # Clip values to 0-255
            x = np.clip(x, 0, 255)
            
            # Place solved values back into result image
            for i, (y, x_coord) in enumerate(self.coords):
                result[y, x_coord, channel] = x[i]

        return result

# --- Usage Example ---
if __name__ == "__main__":
    
    src = cv2.imread("plane.jpg") 
    dst = cv2.imread("sky.jpg")
    
    # Let's assume we already aligned them or resized them roughly
    # For this script, I'll resize src to match dst to avoid index errors
    src = cv2.resize(src, (dst.shape[1], dst.shape[0]))

    mask = np.zeros(dst.shape[:2], dtype=np.uint8)
    h, w = dst.shape[:2]
    mask[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)] = 255

    editor = PoissonEditor(src, dst, mask)
    output = editor.blend()

    cv2.imshow("Original Source", src)
    cv2.imshow("Original Dest", dst)
    cv2.imshow("Poisson Result (Manual)", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()