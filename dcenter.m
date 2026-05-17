function derivative = dcenter(data, dt); 

derivative = [(data(2,:)-data(1,:))/dt;...
    (data(3:end,:) - data(1:end-2,:))/(2*dt);
    (data(end,:)-data(end-1,:))/dt]; 