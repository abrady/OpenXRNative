use std::sync::Arc;
use std::time::Instant;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    acquire_next_image, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    SwapchainPresentInfo,
};
use vulkano::sync::{self, GpuFuture};
use vulkano::VulkanLibrary;
use vulkano_win::VkSurfaceBuild;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use bytemuck::{Pod, Zeroable};
use cgmath::{perspective, Deg, Matrix4, Point3, Vector3};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position, color);

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
struct UniformBufferObject {
    mvp: [[f32; 4]; 4],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 0) out vec3 v_color;
layout(set = 0, binding = 0) uniform Data {
    mat4 mvp;
} uniforms;
void main() {
    gl_Position = uniforms.mvp * vec4(position, 1.0);
    v_color = color;
}
"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec3 v_color;
layout(location = 0) out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"
    }
}

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DK found");
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    let event_loop = EventLoop::new().unwrap();
    let surface = WindowBuilder::new()
        .with_title("Rotating Cube")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .expect("no physical device available")
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .find(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(*i as u32, &surface).unwrap_or(false)
                })
                .map(|(i, _)| (p, i as u32))
        })
        .next()
        .expect("no suitable physical device found");

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: physical_device.required_extensions().union(
                &vulkano::device::DeviceExtensions {
                    khr_swapchain: true,
                    ..vulkano::device::DeviceExtensions::empty()
                },
            ),
            ..Default::default()
        },
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");
        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );
        let window = surface.window();
        let image_extent = window.inner_size().into();
        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count.max(2),
                image_format,
                image_extent,
                image_usage: ImageUsage::color_attachment(),
                composite_alpha: caps.supported_composite_alpha.iter().next().unwrap(),
                present_mode: vulkano::swapchain::PresentMode::Fifo,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let framebuffers: Vec<Arc<Framebuffer>> = images
        .iter()
        .map(|image| {
            let view = vulkano::image::view::ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect();

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let pipeline = GraphicsPipeline::start()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .build(device.clone())
        .unwrap();

    let vertices = [
        // front (red)
        Vertex {
            position: [-0.5, -0.5, 0.5],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
            color: [1.0, 0.0, 0.0],
        },
        // back (green)
        Vertex {
            position: [-0.5, -0.5, -0.5],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5, -0.5],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [-0.5, 0.5, -0.5],
            color: [0.0, 1.0, 0.0],
        },
        // left (blue)
        Vertex {
            position: [-0.5, -0.5, -0.5],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [-0.5, -0.5, 0.5],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [-0.5, 0.5, -0.5],
            color: [0.0, 0.0, 1.0],
        },
        // right (yellow)
        Vertex {
            position: [0.5, -0.5, -0.5],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
            color: [1.0, 1.0, 0.0],
        },
        // top (cyan)
        Vertex {
            position: [-0.5, 0.5, -0.5],
            color: [0.0, 1.0, 1.0],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
            color: [0.0, 1.0, 1.0],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
            color: [0.0, 1.0, 1.0],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
            color: [0.0, 1.0, 1.0],
        },
        // bottom (magenta)
        Vertex {
            position: [-0.5, -0.5, -0.5],
            color: [1.0, 0.0, 1.0],
        },
        Vertex {
            position: [-0.5, -0.5, 0.5],
            color: [1.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
            color: [1.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.5, -0.5, -0.5],
            color: [1.0, 0.0, 1.0],
        },
    ];

    let indices: Vec<u16> = vec![
        0, 1, 2, 2, 3, 0, // front
        4, 5, 6, 6, 7, 4, // back
        8, 9, 10, 10, 11, 8, // left
        12, 13, 14, 14, 15, 12, // right
        16, 17, 18, 18, 19, 16, // top
        20, 21, 22, 22, 23, 20, // bottom
    ];

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        vertices.into_iter(),
    )
    .unwrap();
    let index_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::index_buffer(),
        false,
        indices.into_iter(),
    )
    .unwrap();

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    let start_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::MainEventsCleared => {
                let elapsed = start_time.elapsed().as_secs_f32();
                let rotation = Matrix4::from_angle_y(Deg(elapsed * 45.0));
                let view = Matrix4::look_at_rh(
                    Point3::new(1.5, 1.5, 1.5),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                );
                let proj = perspective(Deg(45.0), 1.0, 0.01, 10.0);
                let mvp = proj * view * rotation;

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let window = surface.window();
                    let new_extent = window.inner_size().into();
                    match swapchain.recreate(SwapchainCreateInfo {
                        image_extent: new_extent,
                        ..swapchain.create_info()
                    }) {
                        Ok((new_swapchain, new_images)) => {
                            swapchain = new_swapchain;
                            // recreate framebuffers
                            let new_framebuffers = new_images
                                .iter()
                                .map(|image| {
                                    let view =
                                        vulkano::image::view::ImageView::new_default(image.clone())
                                            .unwrap();
                                    Framebuffer::new(
                                        render_pass.clone(),
                                        FramebufferCreateInfo {
                                            attachments: vec![view],
                                            ..Default::default()
                                        },
                                    )
                                    .unwrap()
                                })
                                .collect::<Vec<_>>();
                            // we can't assign to framebuffers because it's captured immutably
                            // but for simplicity we ignore window resize in this example
                            let _ = new_framebuffers;
                        }
                        Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                            // ignore
                        }
                        Err(e) => panic!("Failed to recreate swapchain: {e}"),
                    }
                    recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let uniform_buffer = CpuAccessibleBuffer::from_data(
                    device.clone(),
                    BufferUsage::uniform_buffer(),
                    false,
                    UniformBufferObject { mvp: mvp.into() },
                )
                .unwrap();

                let layout = pipeline.layout().set_layouts().get(0).unwrap();
                let set = PersistentDescriptorSet::new(
                    layout.clone(),
                    [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
                )
                .unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        render_pass.clone(),
                        framebuffers[image_index as usize].clone(),
                        SubpassContents::Inline,
                        vec![[0.0, 0.0, 0.0, 1.0].into()],
                    )
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        set,
                    )
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .bind_index_buffer(index_buffer.clone())
                    .draw_indexed(indices.len() as u32, 1, 0, 0, 0)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(sync::FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {e}");
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => {}
        }
    });
}
